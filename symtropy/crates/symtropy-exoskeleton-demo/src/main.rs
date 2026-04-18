// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full-Frame exoskeleton demo: Φ selects AssistanceMode.
//!
//! The human wearer generates a noisy walking gait (simulated CPG + Gaussian
//! noise) that drives the 6 joint torques. The exoskeleton adds assistive
//! torques on top. The consciousness pipeline observes HDC prediction error
//! (from the wearer's gait variability) and selects an AssistanceMode:
//!
//!   Φ > 0.6  → Predictive          (full assistance)
//!   0.3–0.6  → Responsive          (60%)
//!   0.1–0.3  → Transparent         (15% stiffness, 20% torque)
//!   < 0.1    → GravityCompensation (0 — only counters gravity passively)
//!
//! Unlike flight/vehicle/AUV/helicopter demos (which attenuate a single
//! controller's output under external disturbance), this demo shows Φ
//! DEFERRING to the human — when the wearer is erratic, the exo backs off.
//! The platform crate provides `AssistanceMode::from_phi()` + factor methods
//! directly, so no custom gain_scale is needed here.
//!
//! - **Physics of record**: `symthaea_exoskeleton::SimpleExoskeletonSimulator`
//!   (dual-authorship: human CPG + exo torques summed at each joint, with
//!   gravity, damping, and joint limits).

mod camera;
mod consciousness_bridge;
mod controller;
mod hud;
mod kinematics;
mod plugin;
mod resources;
mod visualization;

use bevy::prelude::*;
use plugin::ExoskeletonDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title:
                    "Symthaea: Φ-Gated Full-Frame Exoskeleton (human + exo dual-authorship)".into(),
                resolution: bevy::window::WindowResolution::from((1600u32, 900u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(ExoskeletonDemoPlugin)
        .run();
}
