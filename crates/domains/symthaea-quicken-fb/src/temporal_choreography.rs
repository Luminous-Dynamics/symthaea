// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure temporal choreography policy for Spore Boot Ecology v0.3.3.
//!
//! This module converts already-authoritative stage progress into normalized
//! presentation motion. It owns no clock, performs no I/O, changes no stage
//! duration/order, and cannot infer boot completion, health, readiness, Last
//! Known Good, DRM handoff authority, or physical activation.
//!
//! The important contract is that choreography is parameterized by *semantic
//! stage progress*, not by an invented animation clock. A short real stage and
//! a long real stage therefore traverse the same narrative arc at different
//! wall-clock speeds without the renderer pretending that boot progressed.

use symthaea_boot_ecology::BootStageKind;

use crate::visual_composition::VisualProfile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionIntent {
    Still,
    Relight,
    Germinate,
    Grow,
    Weave,
    Repair,
    GenerationRing,
    HardwareBud,
    Retract,
    Connect,
    Settle,
    Dissolve,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalChoreography {
    pub intent: MotionIntent,
    /// Monotonic semantic phase in `[0, 1]`.
    pub narrative_phase: f32,
    /// Normalized periodic phase for subtle ambient motion.
    pub ambient_phase: f32,
    /// Stage-local emphasis envelope in `[0, 1]`.
    pub hero_envelope: f32,
    /// Relative secondary motion energy in `[0, 1]`.
    pub ambient_gain: f32,
    /// Relative damping/restraint in `[0, 1]`; larger means quieter motion.
    pub damping: f32,
}

impl TemporalChoreography {
    pub fn derive(stage: BootStageKind, stage_progress: f32, profile: VisualProfile) -> Self {
        let progress = finite_unit(stage_progress);
        let narrative_phase = smoothstep(progress);

        let (intent, cycles, hero_envelope, ambient_gain, damping) = match stage {
            BootStageKind::Blackout => (MotionIntent::Still, 0.0, 0.0, 0.0, 1.0),
            BootStageKind::DormantCore => (MotionIntent::Still, 0.55, 0.28, 0.12, 0.88),
            BootStageKind::Relight => (
                MotionIntent::Relight,
                1.10,
                narrative_phase,
                0.26,
                0.58,
            ),
            BootStageKind::Germinate => (
                MotionIntent::Germinate,
                0.85,
                ease_out(progress),
                0.20,
                0.48,
            ),
            BootStageKind::Grow => (
                MotionIntent::Grow,
                1.35,
                narrative_phase,
                0.34,
                0.30,
            ),
            BootStageKind::Anastomose => (
                MotionIntent::Weave,
                1.55,
                narrative_phase,
                0.32,
                0.34,
            ),
            BootStageKind::Repair => (
                MotionIntent::Repair,
                0.72,
                pulse_once(progress),
                0.15,
                0.68,
            ),
            BootStageKind::GrowthRing => (
                MotionIntent::GenerationRing,
                0.82,
                pulse_once(progress),
                0.16,
                0.64,
            ),
            BootStageKind::HardwareBud => (
                MotionIntent::HardwareBud,
                0.78,
                ease_out(progress),
                0.14,
                0.70,
            ),
            BootStageKind::RetractFailedGrowth => (
                MotionIntent::Retract,
                0.62,
                narrative_phase,
                0.10,
                0.78,
            ),
            BootStageKind::MeshLink => (
                MotionIntent::Connect,
                1.18,
                pulse_once(progress),
                0.24,
                0.52,
            ),
            BootStageKind::Settle => (
                MotionIntent::Settle,
                0.46,
                1.0 - narrative_phase * 0.35,
                0.10,
                0.86,
            ),
            BootStageKind::Handoff => {
                let departure = 1.0 - narrative_phase;
                (
                    MotionIntent::Dissolve,
                    0.20,
                    departure,
                    0.08 * departure,
                    0.92 + narrative_phase * 0.08,
                )
            }
        };

        let ambient_phase = if cycles <= 0.0 {
            0.0
        } else {
            (progress * cycles).fract()
        };

        Self {
            intent,
            narrative_phase,
            ambient_phase: finite_unit(ambient_phase),
            hero_envelope: finite_unit(hero_envelope),
            ambient_gain: finite_unit(ambient_gain * profile.motion_scale()),
            damping: finite_unit(damping),
        }
    }
}

impl VisualProfile {
    fn motion_scale(self) -> f32 {
        match self {
            Self::Calm => 0.58,
            Self::Standard => 1.00,
            Self::Rich => 1.16,
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

fn ease_out(value: f32) -> f32 {
    let value = finite_unit(value);
    1.0 - (1.0 - value).powi(3)
}

fn pulse_once(value: f32) -> f32 {
    let value = finite_unit(value);
    (std::f32::consts::PI * value).sin().max(0.0)
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

    #[test]
    fn every_scalar_is_finite_and_bounded() {
        for stage in STAGES {
            for profile in PROFILES {
                for progress in [0.0, 0.2, 0.5, 0.8, 1.0, f32::NAN, f32::INFINITY] {
                    let c = TemporalChoreography::derive(stage, progress, profile);
                    for value in [
                        c.narrative_phase,
                        c.ambient_phase,
                        c.hero_envelope,
                        c.ambient_gain,
                        c.damping,
                    ] {
                        assert!(value.is_finite(), "{stage:?} {profile:?}: {value}");
                        assert!((0.0..=1.0).contains(&value));
                    }
                }
            }
        }
    }

    #[test]
    fn narrative_phase_is_monotonic_and_semantic() {
        for stage in STAGES {
            let mut previous = 0.0;
            for progress in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
                let c = TemporalChoreography::derive(stage, progress, VisualProfile::Standard);
                assert!(c.narrative_phase + f32::EPSILON >= previous);
                previous = c.narrative_phase;
            }
        }
    }

    #[test]
    fn semantic_intent_does_not_change_with_profile() {
        for stage in STAGES {
            let calm = TemporalChoreography::derive(stage, 0.4, VisualProfile::Calm);
            let standard = TemporalChoreography::derive(stage, 0.4, VisualProfile::Standard);
            let rich = TemporalChoreography::derive(stage, 0.4, VisualProfile::Rich);
            assert_eq!(calm.intent, standard.intent);
            assert_eq!(standard.intent, rich.intent);
            assert_eq!(calm.narrative_phase, standard.narrative_phase);
            assert_eq!(standard.narrative_phase, rich.narrative_phase);
            assert!(calm.ambient_gain <= standard.ambient_gain + f32::EPSILON);
            assert!(standard.ambient_gain <= rich.ambient_gain + f32::EPSILON);
        }
    }

    #[test]
    fn recovery_is_quieter_than_growth() {
        let grow = TemporalChoreography::derive(
            BootStageKind::Grow,
            0.5,
            VisualProfile::Standard,
        );
        let repair = TemporalChoreography::derive(
            BootStageKind::Repair,
            0.5,
            VisualProfile::Standard,
        );
        let rollback = TemporalChoreography::derive(
            BootStageKind::RetractFailedGrowth,
            0.5,
            VisualProfile::Standard,
        );
        assert!(repair.ambient_gain < grow.ambient_gain);
        assert!(rollback.ambient_gain < repair.ambient_gain);
        assert!(repair.damping > grow.damping);
        assert!(rollback.damping > repair.damping);
    }

    #[test]
    fn persistent_events_have_distinct_motion_intents() {
        assert_eq!(
            TemporalChoreography::derive(
                BootStageKind::GrowthRing,
                0.5,
                VisualProfile::Standard,
            )
            .intent,
            MotionIntent::GenerationRing,
        );
        assert_eq!(
            TemporalChoreography::derive(
                BootStageKind::HardwareBud,
                0.5,
                VisualProfile::Standard,
            )
            .intent,
            MotionIntent::HardwareBud,
        );
        assert_eq!(
            TemporalChoreography::derive(
                BootStageKind::MeshLink,
                0.5,
                VisualProfile::Standard,
            )
            .intent,
            MotionIntent::Connect,
        );
        assert_eq!(
            TemporalChoreography::derive(
                BootStageKind::RetractFailedGrowth,
                0.5,
                VisualProfile::Standard,
            )
            .intent,
            MotionIntent::Retract,
        );
    }

    #[test]
    fn handoff_dissolves_motion_as_well_as_pixels() {
        let mut previous_hero = f32::INFINITY;
        let mut previous_ambient = f32::INFINITY;
        let mut previous_damping = 0.0;
        for progress in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let c = TemporalChoreography::derive(
                BootStageKind::Handoff,
                progress,
                VisualProfile::Rich,
            );
            assert_eq!(c.intent, MotionIntent::Dissolve);
            assert!(c.hero_envelope <= previous_hero + f32::EPSILON);
            assert!(c.ambient_gain <= previous_ambient + f32::EPSILON);
            assert!(c.damping + f32::EPSILON >= previous_damping);
            previous_hero = c.hero_envelope;
            previous_ambient = c.ambient_gain;
            previous_damping = c.damping;
        }

        let end = TemporalChoreography::derive(
            BootStageKind::Handoff,
            1.0,
            VisualProfile::Rich,
        );
        assert_eq!(end.hero_envelope, 0.0);
        assert_eq!(end.ambient_gain, 0.0);
        assert_eq!(end.damping, 1.0);
    }

    #[test]
    fn invalid_progress_fails_closed() {
        let c = TemporalChoreography::derive(
            BootStageKind::Grow,
            f32::NAN,
            VisualProfile::Rich,
        );
        assert_eq!(c.narrative_phase, 0.0);
        assert_eq!(c.ambient_phase, 0.0);
    }
}
