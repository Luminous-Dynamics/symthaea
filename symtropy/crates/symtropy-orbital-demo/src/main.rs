// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Orbital-arm deployment demo: Φ tracks orbital-phase constraints.
//!
//! A 7-joint spacecraft manipulator arm tries to reach a payload-handoff
//! pose in space. The spacecraft tumbles from Newton-3rd-law reaction
//! torques as the arm moves; reaction wheels desaturate slowly. An orbital
//! phase (90-min LEO) drives `solar_exposure` and `comm_window` — the
//! simulator already owns this.
//!
//! Narrative difference from prior demos:
//! - Flight / vehicle / AUV / helicopter attenuate motor authority under
//!   EXTERNAL disturbance (wind, ice, current, gusts).
//! - Exoskeleton uses Φ to DEFER to the human (mode selection).
//! - **Orbital uses Φ to track MISSION-phase constraints** — aggressive
//!   deployment only when comm window is open, solar power available, and
//!   attitude drift bounded. Low Φ → hold the pose, let reaction wheels
//!   catch up.
//!
//! - **Physics of record**: `symthaea_orbital::SimpleOrbitalSimulator`
//!   (arm torques, reaction-wheel desaturation, orbital-phase solar +
//!   comm-window sweep).

mod camera;
mod consciousness_bridge;
mod controller;
mod hud;
mod plugin;
mod resources;
mod visualization;

use bevy::prelude::*;
use plugin::OrbitalDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea: Φ-Gated Orbital Arm (comm-window + solar + attitude)".into(),
                resolution: bevy::window::WindowResolution::from((1600u32, 900u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(OrbitalDemoPlugin)
        .add_plugins(symtropy_demo_capture::CapturePlugin::new("orbital"))
        .run();
}
