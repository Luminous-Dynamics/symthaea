// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic demo director for Sol Atlas.
//!
//! Drives camera, timeline, and aesthetics through a scripted cinematic
//! narrative. Combined with frame_capture (--record), produces a demo video.
//!
//! Future: wire Symthaea FEP agent to modulate within each phase.

use bevy::prelude::*;
use sol_atlas_bevy::camera::OrbitalCameraConfig;
use sol_atlas_bevy::timeline::TimelineState;
use symthaea_fep::EnhancedFEPBridge;
use symthaea_fep::MotorCommandType;

use crate::systems::atlas::CurrentAesthetic;

/// Demo narrative phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DemoPhase {
    /// 0-5s: orbit view, holographic aesthetic, auto-drift
    Opening,
    /// 5-10s: zoom into Europe/Middle East, markers appear
    ZoomIn,
    /// 10-18s: scrub timeline, EROI decline visible
    TimelineScrub,
    /// 18-22s: cycle aesthetics (Satellite → Night → Holographic)
    AestheticCycle,
    /// 22-26s: pull back to orbit, LOD transition
    ZoomOut,
    /// 26-30s: slow drift, clean closing shot
    Closing,
    /// Demo complete
    Done,
}

impl DemoPhase {
    fn duration(&self) -> f32 {
        match self {
            Self::Opening => 5.0,
            Self::ZoomIn => 5.0,
            Self::TimelineScrub => 8.0,
            Self::AestheticCycle => 4.0,
            Self::ZoomOut => 4.0,
            Self::Closing => 4.0,
            Self::Done => 0.0,
        }
    }

    fn next(&self) -> Self {
        match self {
            Self::Opening => Self::ZoomIn,
            Self::ZoomIn => Self::TimelineScrub,
            Self::TimelineScrub => Self::AestheticCycle,
            Self::AestheticCycle => Self::ZoomOut,
            Self::ZoomOut => Self::Closing,
            Self::Closing => Self::Done,
            Self::Done => Self::Done,
        }
    }
}

/// Demo director resource.
#[derive(Resource)]
pub struct DemoDirector {
    pub enabled: bool,
    pub phase: DemoPhase,
    pub phase_timer: f32,
    pub total_time: f32,
    /// FEP agent for consciousness-driven modulation (None = fully deterministic).
    pub fep_bridge: Option<EnhancedFEPBridge>,
    /// How much the FEP agent modulates the deterministic path (0.0 = none, 1.0 = full).
    pub modulation_strength: f32,
    /// Last prediction error from the FEP agent.
    pub last_prediction_error: f64,
    /// Phi (consciousness integration) of the demo.
    pub phi: f64,
}

impl Default for DemoDirector {
    fn default() -> Self {
        Self {
            enabled: false,
            phase: DemoPhase::Opening,
            phase_timer: 0.0,
            total_time: 0.0,
            fep_bridge: None, // deterministic by default
            modulation_strength: 0.0,
            last_prediction_error: 0.0,
            phi: 0.0,
        }
    }
}

/// Deterministic demo direction system.
/// Smooth interpolation, no FEP agent — guaranteed cinematic output.
pub fn demo_director_system(
    time: Res<Time>,
    mut director: ResMut<DemoDirector>,
    mut camera: ResMut<OrbitalCameraConfig>,
    mut timeline: ResMut<TimelineState>,
    mut aesthetic: ResMut<CurrentAesthetic>,
) {
    if !director.enabled || director.phase == DemoPhase::Done {
        return;
    }

    let dt = time.delta_secs();
    director.phase_timer += dt;
    director.total_time += dt;

    // Progress within current phase (0.0 to 1.0)
    let t = (director.phase_timer / director.phase.duration()).clamp(0.0, 1.0);
    // Smooth easing (cubic ease-in-out)
    let ease = if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    };

    match director.phase {
        DemoPhase::Opening => {
            // Orbit view, slow drift, holographic
            camera.auto_rotate = true;
            camera.distance = 4.5;
            // Gentle theta drift
            camera.theta += 0.0005;
        }

        DemoPhase::ZoomIn => {
            // Zoom toward Europe/Middle East (theta ≈ 0.5, phi ≈ 0.3)
            camera.auto_rotate = false;
            camera.distance = 4.5 - ease * 1.0; // 4.5 → 3.5 (stays in Atmosphere — clean, no markers)
            camera.theta += (0.8 - camera.theta) * 0.04; // more rotation to show coverage
            camera.phi += (0.2 - camera.phi) * 0.03;
            // Start timeline at year 50
            timeline.year = (ease * 50.0) as u32;
        }

        DemoPhase::TimelineScrub => {
            // Hold camera, scrub timeline from 50 to 200
            camera.distance += (2.8 - camera.distance) * 0.02;
            timeline.year = 50 + (ease * 150.0) as u32; // 50 → 200
                                                        // Slow rotation during scrub
            camera.theta += 0.0003;
        }

        DemoPhase::AestheticCycle => {
            // Cycle through aesthetics
            let phase_third = t * 3.0;
            let new_aesthetic = if phase_third < 1.0 {
                sol_atlas_core::aesthetics::Aesthetic::Satellite
            } else if phase_third < 2.0 {
                sol_atlas_core::aesthetics::Aesthetic::Night
            } else {
                sol_atlas_core::aesthetics::Aesthetic::Holographic
            };
            if new_aesthetic != aesthetic.aesthetic {
                aesthetic.aesthetic = new_aesthetic;
                aesthetic.changed = true;
            }
            // Pull back slightly
            camera.distance += (4.0 - camera.distance) * 0.03;
            camera.theta += 0.001;
        }

        DemoPhase::ZoomOut => {
            // Pull back to orbit
            camera.distance += (5.0 - camera.distance) * 0.04;
            camera.auto_rotate = true;
            // Ensure holographic
            if aesthetic.aesthetic != sol_atlas_core::aesthetics::Aesthetic::Holographic {
                aesthetic.aesthetic = sol_atlas_core::aesthetics::Aesthetic::Holographic;
                aesthetic.changed = true;
            }
        }

        DemoPhase::Closing => {
            // Slow drift, clean shot
            camera.distance += (4.5 - camera.distance) * 0.02;
            camera.theta += 0.0004;
        }

        DemoPhase::Done => {}
    }

    // FEP modulation — consciousness-driven adjustments to the deterministic path
    let mod_strength = director.modulation_strength;
    let last_pe = director.last_prediction_error;
    let phi = director.phi;
    if mod_strength > 0.0 {
        if let Some(ref mut bridge) = director.fep_bridge {
            let phase_progress = t as f64;
            let cam_dist_normalized = (camera.distance as f64 - 1.8) / (8.0 - 1.8);
            let coherence = 1.0 - last_pe;

            let result = bridge.cycle(phi, phase_progress, coherence, cam_dist_normalized);

            let new_pe = result.fep_result.free_energy.abs().min(1.0);
            let new_phi = coherence * 0.7 + phase_progress * 0.3;
            let intensity = result.motor_command.intensity as f32;

            match result.motor_command.command_type {
                MotorCommandType::AttentionShift => {
                    camera.theta += intensity * 0.01 * mod_strength;
                }
                MotorCommandType::ExplorationTrigger => {
                    camera.distance -= intensity * 0.02 * mod_strength;
                }
                MotorCommandType::ReflectionInitiate => {
                    timeline.speed = 10.0 * (1.0 - intensity * 0.5 * mod_strength);
                }
                _ => {}
            }

            // Update director state after borrow is released
            drop(bridge);
            director.last_prediction_error = new_pe;
            director.phi = new_phi;
        }
    }

    // Phase transition
    if director.phase_timer >= director.phase.duration() {
        let old = director.phase;
        director.phase = director.phase.next();
        director.phase_timer = 0.0;
        if director.phase != DemoPhase::Done {
            info!(
                "[demo] Phase: {:?} → {:?} (t={:.1}s)",
                old, director.phase, director.total_time
            );
        } else {
            info!("[demo] Complete! {:.1}s total", director.total_time);
        }
    }
}
