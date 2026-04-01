// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic demo director for Terra Atlas.
//!
//! Drives camera, timeline, and aesthetics through a scripted cinematic
//! narrative. Combined with frame_capture (--record), produces a demo video.
//!
//! Future: wire Symthaea FEP agent to modulate within each phase.

use bevy::prelude::*;
use terra_atlas_bevy::camera::OrbitalCameraConfig;
use terra_atlas_bevy::timeline::TimelineState;

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
}

impl Default for DemoDirector {
    fn default() -> Self {
        Self {
            enabled: false,
            phase: DemoPhase::Opening,
            phase_timer: 0.0,
            total_time: 0.0,
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
    let ease = if t < 0.5 { 4.0 * t * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(3) / 2.0 };

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
            camera.distance = 4.5 - ease * 2.0; // 4.5 → 2.5
            camera.theta += (0.5 - camera.theta) * 0.03;
            camera.phi += (0.3 - camera.phi) * 0.03;
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
                terra_atlas_core::aesthetics::Aesthetic::Satellite
            } else if phase_third < 2.0 {
                terra_atlas_core::aesthetics::Aesthetic::Night
            } else {
                terra_atlas_core::aesthetics::Aesthetic::Holographic
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
            if aesthetic.aesthetic != terra_atlas_core::aesthetics::Aesthetic::Holographic {
                aesthetic.aesthetic = terra_atlas_core::aesthetics::Aesthetic::Holographic;
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

    // Phase transition
    if director.phase_timer >= director.phase.duration() {
        let old = director.phase;
        director.phase = director.phase.next();
        director.phase_timer = 0.0;
        if director.phase != DemoPhase::Done {
            info!("[demo] Phase: {:?} → {:?} (t={:.1}s)", old, director.phase, director.total_time);
        } else {
            info!("[demo] Complete! {:.1}s total", director.total_time);
        }
    }
}
