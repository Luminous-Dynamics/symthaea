// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Surgical-robot demo: Φ selects SurgicalSafetyLevel; cautery is a hard interlock.
//!
//! A 6-DOF surgical manipulator operates near a target tissue, with a critical
//! structure (nerve/vessel) in the field and a trocar port constraining lateral
//! motion. The consciousness pipeline observes tissue proximity, trocar
//! compliance, tip force, and HDC PE; the platform crate maps Φ onto:
//!
//!   Φ > 0.6  → FullControl  (100% torque, cautery allowed)
//!   0.3–0.6  → Reduced      (40% torque, cautery BLOCKED)
//!   0.1–0.3  → Freeze       (0% torque, cautery BLOCKED)
//!   < 0.1    → Retract      (0% torque, cautery BLOCKED)
//!
//! `SurgicalSafetyLevel::cautery_allowed()` is a HARD INTERLOCK: only
//! FullControl permits energy delivery. This is the first demo in the series
//! where a non-magnitude channel (cautery power) is gated binary-by-mode
//! rather than scaled continuously — the consciousness-as-safety-layer story
//! in its sharpest form.
//!
//! - **Physics of record**: `symthaea_surgical::SimpleSurgicalSimulator`
//!   (joint dynamics, tremor filter, FK to tip, tissue contact, trocar
//!   compliance).

mod camera;
mod consciousness_bridge;
mod controller;
mod hud;
mod plugin;
mod resources;
mod visualization;

use bevy::prelude::*;
use plugin::SurgicalDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea: Φ-Gated Surgical Robot (proximity + cautery interlock)".into(),
                resolution: bevy::window::WindowResolution::from((1600u32, 900u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(SurgicalDemoPlugin)
        .add_plugins(symtropy_demo_capture::CapturePlugin::new("surgical"))
        .run();
}
