// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! AUV waypoint demo: Φ-gated thruster authority under underwater currents.
//!
//! The AUV navigates a 3D waypoint path at varying depths. A slowly-rotating
//! underwater current (external force) pushes it laterally. The consciousness
//! pipeline observes HDC prediction error + current magnitude, returning a
//! motor gain that attenuates thruster authority when the disturbance dominates.
//!
//! - **Physics of record**: `symthaea_auv::SimpleAuvSimulator` (6DOF
//!   hydrodynamics: added mass, quadratic drag, buoyancy).
//! - **Consciousness side-channel**: `RoboticAgent` (PlatformType::Auv).
//! - **Control**: depth/heading/surge PIDs mapped to the 8-thruster vector.

mod camera;
mod consciousness_bridge;
mod controller;
mod current;
mod hud;
mod plugin;
mod resources;
mod visualization;

use bevy::prelude::*;
use plugin::AuvDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea: Φ-Gated AUV Waypoint Navigation (underwater currents)".into(),
                resolution: bevy::window::WindowResolution::from((1600u32, 900u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(AuvDemoPlugin)
        .add_plugins(symtropy_demo_capture::CapturePlugin::new("auv"))
        .run();
}
