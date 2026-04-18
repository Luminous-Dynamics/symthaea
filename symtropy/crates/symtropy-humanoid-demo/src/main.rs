// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bipedal humanoid balance demo: Φ attenuates balance-control torque
//! authority under periodic push perturbations.
//!
//! A 21-DOF DMC21-morphology humanoid stands in place. Periodic horizontal
//! pushes strike the torso. The controller tries to hold the default
//! standing pose via joint-space PD; the consciousness pipeline observes
//! uprightness + root height + angular-velocity magnitude + HDC PE, and
//! returns a motor gain that attenuates all 21 torques uniformly.
//!
//! - **Physics of record**: `symthaea_humanoid::SimpleHumanoidSimulator`
//!   (rich anatomical body model, ground contact, CoM tracking).
//! - **Visualization**: no hand-rolled skeleton FK — the simulator already
//!   computes extremity world positions (hands + feet), so the demo just
//!   places spheres at those positions and connects them to the torso with
//!   simple oriented cylinders.

mod camera;
mod consciousness_bridge;
mod controller;
mod hud;
mod plugin;
mod push;
mod resources;
mod visualization;

use bevy::prelude::*;
use plugin::HumanoidDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea: Φ-Gated Bipedal Humanoid Balance (push perturbations)".into(),
                resolution: bevy::window::WindowResolution::from((1600u32, 900u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(HumanoidDemoPlugin)
        .add_plugins(symtropy_demo_capture::CapturePlugin::new("humanoid"))
        .run();
}
