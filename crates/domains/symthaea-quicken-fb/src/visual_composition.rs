// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure perceptual-composition policy for Spore Boot Ecology v0.3.3.
//!
//! This module coordinates renderer attention only. It performs no I/O, owns no
//! boot state, mutates no `BootGenome`, and cannot influence health, Last Known
//! Good, handoff readiness, DRM ownership, or physical activation.

use symthaea_boot_ecology::BootStageKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisualProfile {
    Calm,
    #[default]
    Standard,
    Rich,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualHero {
    Substrate,
    Topology,
    Relight,
    Repair,
    Generation,
    Hardware,
    Rollback,
    Mesh,
    Handoff,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualCompositionBudget {
    pub hero: VisualHero,
    pub topology: f32,
    pub accent: f32,
    pub mesh: f32,
    pub holography: f32,
    pub membrane: f32,
    pub caustics: f32,
    pub bloom: f32,
    pub identity: f32,
}

impl VisualCompositionBudget {
    /// Derive a bounded presentation budget from already-authoritative semantic
    /// presentation inputs. `stage_intensity` is visual energy only; it is not a
    /// confidence, health, readiness, or completion signal.
    pub fn derive(
        stage: BootStageKind,
        stage_progress: f32,
        stage_intensity: f32,
        profile: VisualProfile,
    ) -> Self {
        let progress = finite_unit(stage_progress);
        let intensity = finite_unit(stage_intensity);
        let semantic_energy = 0.25 + intensity * 0.75;

        let mut budget = match stage {
            BootStageKind::Blackout => Self::zero(VisualHero::Substrate),
            BootStageKind::DormantCore => Self::from_layers(
                VisualHero::Topology,
                0.95,
                [0.24, 0.00, 0.15, 0.25, 0.02, 0.12, 0.55],
            ),
            BootStageKind::Relight => Self::from_layers(
                VisualHero::Relight,
                1.00,
                [0.50, 0.08, 0.45, 0.32, 0.05, 0.22, 0.55],
            ),
            BootStageKind::Germinate => Self::from_layers(
                VisualHero::Topology,
                1.00,
                [0.40, 0.00, 0.18, 0.32, 0.04, 0.18, 0.52],
            ),
            BootStageKind::Grow => Self::from_layers(
                VisualHero::Topology,
                1.00,
                [0.32, 0.08, 0.28, 0.20, 0.06, 0.18, 0.52],
            ),
            BootStageKind::Anastomose => Self::from_layers(
                VisualHero::Topology,
                1.00,
                [0.38, 0.20, 0.28, 0.18, 0.05, 0.17, 0.52],
            ),
            BootStageKind::Repair => Self::from_layers(
                VisualHero::Repair,
                1.00,
                [0.78, 0.04, 0.12, 0.24, 0.05, 0.24, 0.58],
            ),
            BootStageKind::GrowthRing => Self::from_layers(
                VisualHero::Generation,
                1.00,
                [0.82, 0.05, 0.16, 0.18, 0.03, 0.24, 0.56],
            ),
            BootStageKind::HardwareBud => Self::from_layers(
                VisualHero::Hardware,
                1.00,
                [0.64, 0.05, 0.18, 0.18, 0.04, 0.18, 0.55],
            ),
            BootStageKind::RetractFailedGrowth => Self::from_layers(
                VisualHero::Rollback,
                1.00,
                [0.76, 0.02, 0.10, 0.20, 0.04, 0.20, 0.60],
            ),
            BootStageKind::MeshLink => Self::from_layers(
                VisualHero::Mesh,
                1.00,
                [0.42, 0.78, 0.40, 0.12, 0.02, 0.18, 0.54],
            ),
            BootStageKind::Settle => Self::from_layers(
                VisualHero::Topology,
                0.92,
                [0.18, 0.14, 0.16, 0.16, 0.02, 0.12, 0.50],
            ),
            BootStageKind::Handoff => {
                let departure = 1.0 - smoothstep(progress);
                // v0.3.3 still owns a standalone DRM surface. Without an explicit
                // morphology-transfer consumer downstream, retaining a visible
                // structural seed would merely create a hard cut when DRM is
                // released. Resolve almost entirely to darkness instead. A future
                // continuity handoff must be an explicit, separately tested mode.
                Self::from_layers(
                    VisualHero::Handoff,
                    0.008 + 0.892 * departure,
                    [
                        0.18 * departure,
                        0.10 * departure,
                        0.18 * departure,
                        0.14 * departure,
                        0.02 * departure,
                        0.10 * departure,
                        0.42 * departure,
                    ],
                )
            }
        };

        if stage != BootStageKind::Blackout {
            budget.scale_secondaries(semantic_energy * profile.secondary_scale());
            budget.identity *= profile.identity_scale();
        }
        budget.clamp();
        budget
    }

    pub fn secondary_sum(self) -> f32 {
        self.accent
            + self.mesh
            + self.holography
            + self.membrane
            + self.caustics
            + self.bloom
            + self.identity
    }

    pub fn should_render(gain: f32) -> bool {
        gain.is_finite() && gain >= 0.01
    }

    fn from_layers(hero: VisualHero, topology: f32, layers: [f32; 7]) -> Self {
        let [accent, mesh, holography, membrane, caustics, bloom, identity] = layers;
        Self {
            hero,
            topology,
            accent,
            mesh,
            holography,
            membrane,
            caustics,
            bloom,
            identity,
        }
    }

    fn zero(hero: VisualHero) -> Self {
        Self::from_layers(hero, 0.0, [0.0; 7])
    }

    fn scale_secondaries(&mut self, scale: f32) {
        self.accent *= scale;
        self.mesh *= scale;
        self.holography *= scale;
        self.membrane *= scale;
        self.caustics *= scale;
        self.bloom *= scale;
    }

    fn clamp(&mut self) {
        self.topology = finite_unit(self.topology);
        self.accent = finite_unit(self.accent).min(self.topology);
        self.mesh = finite_unit(self.mesh).min(self.topology);
        self.holography = finite_unit(self.holography).min(self.topology);
        self.membrane = finite_unit(self.membrane).min(self.topology);
        self.caustics = finite_unit(self.caustics).min(self.topology);
        self.bloom = finite_unit(self.bloom).min(self.topology);
        self.identity = finite_unit(self.identity);
    }
}

impl VisualProfile {
    fn secondary_scale(self) -> f32 {
        match self {
            Self::Calm => 0.55,
            Self::Standard => 1.00,
            Self::Rich => 1.20,
        }
    }

    fn identity_scale(self) -> f32 {
        match self {
            Self::Calm => 0.92,
            Self::Standard | Self::Rich => 1.00,
        }
    }
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn smoothstep(value: f32) -> f32 {
    let value = finite_unit(value);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAGES: [BootStageKind; 13] = [
        BootStageKind::Blackout,
        BootStageKind::DormantCore,
        BootStageKind::Relight,
        BootStageKind::Germinate,
        BootStageKind::Grow,
        BootStageKind::Anastomose,
        BootStageKind::Repair,
        BootStageKind::GrowthRing,
        BootStageKind::HardwareBud,
        BootStageKind::RetractFailedGrowth,
        BootStageKind::MeshLink,
        BootStageKind::Settle,
        BootStageKind::Handoff,
    ];

    const PROFILES: [VisualProfile; 3] = [
        VisualProfile::Calm,
        VisualProfile::Standard,
        VisualProfile::Rich,
    ];

    fn gains(budget: VisualCompositionBudget) -> [f32; 8] {
        [
            budget.topology,
            budget.accent,
            budget.mesh,
            budget.holography,
            budget.membrane,
            budget.caustics,
            budget.bloom,
            budget.identity,
        ]
    }

    #[test]
    fn all_gains_are_finite_and_bounded() {
        for stage in STAGES {
            for profile in PROFILES {
                for progress in [0.0, 0.25, 0.5, 0.75, 1.0, f32::NAN] {
                    for intensity in [0.0, 0.5, 1.0, f32::INFINITY] {
                        let budget = VisualCompositionBudget::derive(
                            stage,
                            progress,
                            intensity,
                            profile,
                        );
                        for gain in gains(budget) {
                            assert!(gain.is_finite(), "{stage:?} {profile:?}: {gain}");
                            assert!((0.0..=1.0).contains(&gain));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn topology_remains_primary_structural_gain() {
        for stage in STAGES {
            if stage == BootStageKind::Blackout {
                continue;
            }
            let budget = VisualCompositionBudget::derive(
                stage,
                0.5,
                1.0,
                VisualProfile::Rich,
            );
            for secondary in [
                budget.accent,
                budget.mesh,
                budget.holography,
                budget.membrane,
                budget.caustics,
                budget.bloom,
            ] {
                assert!(budget.topology >= secondary, "{stage:?}: {budget:?}");
            }
        }
    }

    #[test]
    fn recovery_and_rollback_suppress_unrelated_spectacle() {
        for stage in [BootStageKind::Repair, BootStageKind::RetractFailedGrowth] {
            let budget = VisualCompositionBudget::derive(
                stage,
                0.5,
                1.0,
                VisualProfile::Standard,
            );
            assert!(budget.accent > budget.holography);
            assert!(budget.accent > budget.caustics * 4.0);
            assert!(budget.mesh < budget.accent * 0.10);
        }
    }

    #[test]
    fn update_mesh_and_hardware_have_distinct_attention() {
        let update = VisualCompositionBudget::derive(
            BootStageKind::GrowthRing,
            0.5,
            1.0,
            VisualProfile::Standard,
        );
        let mesh = VisualCompositionBudget::derive(
            BootStageKind::MeshLink,
            0.5,
            1.0,
            VisualProfile::Standard,
        );
        let hardware = VisualCompositionBudget::derive(
            BootStageKind::HardwareBud,
            0.5,
            1.0,
            VisualProfile::Standard,
        );
        assert_eq!(update.hero, VisualHero::Generation);
        assert_eq!(mesh.hero, VisualHero::Mesh);
        assert_eq!(hardware.hero, VisualHero::Hardware);
        assert!(update.accent > update.mesh);
        assert!(mesh.mesh > mesh.accent);
        assert!(hardware.accent > hardware.holography * 3.0);
        assert!(hardware.mesh < hardware.accent * 0.10);
    }

    #[test]
    fn lower_stage_intensity_never_increases_secondary_layers() {
        for stage in STAGES {
            for profile in PROFILES {
                let low = VisualCompositionBudget::derive(stage, 0.5, 0.2, profile);
                let high = VisualCompositionBudget::derive(stage, 0.5, 0.9, profile);
                assert!(low.accent <= high.accent);
                assert!(low.mesh <= high.mesh);
                assert!(low.holography <= high.holography);
                assert!(low.membrane <= high.membrane);
                assert!(low.caustics <= high.caustics);
                assert!(low.bloom <= high.bloom);
            }
        }
    }

    #[test]
    fn profiles_order_secondary_richness_without_changing_semantics() {
        for stage in STAGES {
            let calm = VisualCompositionBudget::derive(stage, 0.5, 0.8, VisualProfile::Calm);
            let standard =
                VisualCompositionBudget::derive(stage, 0.5, 0.8, VisualProfile::Standard);
            let rich = VisualCompositionBudget::derive(stage, 0.5, 0.8, VisualProfile::Rich);
            assert_eq!(calm.hero, standard.hero);
            assert_eq!(standard.hero, rich.hero);
            assert!(calm.secondary_sum() <= standard.secondary_sum() + f32::EPSILON);
            assert!(standard.secondary_sum() <= rich.secondary_sum() + f32::EPSILON);
        }
    }

    #[test]
    fn handoff_every_layer_monotonically_simplifies() {
        let mut previous = [f32::INFINITY; 8];
        for progress in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let budget = VisualCompositionBudget::derive(
                BootStageKind::Handoff,
                progress,
                1.0,
                VisualProfile::Standard,
            );
            let current = gains(budget);
            for (before, after) in previous.into_iter().zip(current) {
                assert!(after <= before + f32::EPSILON);
            }
            previous = current;
        }
    }

    #[test]
    fn standalone_handoff_finishes_below_render_threshold() {
        for profile in PROFILES {
            let final_budget = VisualCompositionBudget::derive(
                BootStageKind::Handoff,
                1.0,
                1.0,
                profile,
            );
            assert_eq!(final_budget.hero, VisualHero::Handoff);
            assert!(final_budget.secondary_sum() <= 0.001);
            assert!(final_budget.topology < 0.01);
            assert!(!VisualCompositionBudget::should_render(final_budget.topology));
        }
    }

    #[test]
    fn blackout_is_exactly_empty() {
        let budget = VisualCompositionBudget::derive(
            BootStageKind::Blackout,
            0.5,
            1.0,
            VisualProfile::Rich,
        );
        assert_eq!(budget.hero, VisualHero::Substrate);
        assert_eq!(gains(budget), [0.0; 8]);
    }

    #[test]
    fn tiny_gains_can_skip_work() {
        assert!(!VisualCompositionBudget::should_render(0.0));
        assert!(!VisualCompositionBudget::should_render(0.009));
        assert!(VisualCompositionBudget::should_render(0.01));
        assert!(!VisualCompositionBudget::should_render(f32::NAN));
    }
}
