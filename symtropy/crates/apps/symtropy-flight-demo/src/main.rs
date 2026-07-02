// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quadrotor waypoint flight demo: Φ-gated motor authority under wind-gust disturbance.
//!
//! The quadrotor flies a figure-8 waypoint pattern. Periodic wind gusts (lateral
//! external forces) stress the controller. The consciousness pipeline observes
//! HDC prediction error and danger level, returning a motor gain that down-weights
//! aggressive control commands when uncertainty is high.
//!
//! - **Physics of record**: `symthaea_multirotor::SimplePhysicsSimulator` (13-D
//!   position + quaternion + velocities, motor lag, aero drag).
//! - **Consciousness side-channel**: `symtropy_robotics_bridge::RoboticAgent`
//!   runs FEP perceive → action → MasterConsciousnessEquation → safety tier.
//! - **Control**: attitude PID with thrust setpoint from altitude/position error,
//!   outputs scaled by the returned motor gain.

mod camera;
mod consciousness_bridge;
mod controller;
mod hud;
mod plugin;
mod resources;
mod visualization;
mod wind;

use bevy::prelude::*;
use plugin::FlightDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea: Φ-Gated Quadrotor Flight (wind-gust disturbance)".into(),
                resolution: bevy::window::WindowResolution::from((1600u32, 900u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FlightDemoPlugin)
        .add_plugins(symtropy_demo_capture::CapturePlugin::new("flight"))
        .run();
}
