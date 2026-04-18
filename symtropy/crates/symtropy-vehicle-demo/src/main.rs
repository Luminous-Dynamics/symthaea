// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Autonomous vehicle waypoint demo: Φ-gated steering/throttle across ice patches.
//!
//! The car drives a figure-8 track at a cruise speed. Periodic ice patches
//! (low-friction zones + sideways gusts) stress the controller. The
//! consciousness pipeline observes HDC prediction error and danger level,
//! returning a motor gain that down-weights aggressive steering/throttle
//! when tire slip rises.
//!
//! - **Physics of record**: `symthaea_vehicle::BicycleModelSimulator`
//!   (Pacejka tire saturation, weight transfer, aerodynamic drag).
//! - **Consciousness side-channel**: `RoboticAgent` (PlatformType::Vehicle).
//! - **Control**: Stanley-style lateral + PI longitudinal, outputs scaled
//!   by the returned motor gain.

mod camera;
mod consciousness_bridge;
mod controller;
mod hud;
mod ice_patch;
mod plugin;
mod resources;
mod visualization;

use bevy::prelude::*;
use plugin::VehicleDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea: Φ-Gated Autonomous Vehicle (ice patches + crosswind)".into(),
                resolution: bevy::window::WindowResolution::from((1600u32, 900u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(VehicleDemoPlugin)
        .add_plugins(symtropy_demo_capture::CapturePlugin::new("vehicle"))
        .run();
}
