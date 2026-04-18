// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SAR-helicopter hover demo: Φ-gated cyclic+collective authority under
//! Dryden wind gusts.
//!
//! The helicopter holds station over a ground survivor marker at ~20 m
//! altitude. A Dryden wind model emits a steady wind + filtered Gaussian
//! gusts that push the aircraft off-station. The consciousness pipeline
//! observes HDC prediction error + wind intensity, returning a motor gain
//! that attenuates aggressive corrections while leaving hover-thrust intact.
//!
//! - **Physics of record**: `symthaea_helicopter::SimpleHelicopterSimulator`
//!   (rotor dynamics, gyroscopic precession, aero drag, 500 kg Robinson
//!   R44-class airframe).
//! - **Consciousness side-channel**: `RoboticAgent` (PlatformType::Helicopter).
//! - **Wind source**: `symthaea_helicopter::wind_model::WindModel` with the
//!   `moderate_wind` preset (5 m/s steady + 6 m/s gust intensity).

mod camera;
mod consciousness_bridge;
mod controller;
mod hud;
mod plugin;
mod resources;
mod visualization;

use bevy::prelude::*;
use plugin::HelicopterDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea: Φ-Gated SAR Helicopter Hover (Dryden wind)".into(),
                resolution: bevy::window::WindowResolution::from((1600u32, 900u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(HelicopterDemoPlugin)
        .add_plugins(symtropy_demo_capture::CapturePlugin::new("helicopter"))
        .run();
}
