// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quadruped demo: Φ selects GaitType on a 4-leg × 3-joint platform.
//!
//! A 12-DoF quadruped walks forward. Periodic internal perturbation
//! spikes the HDC prediction error, which drops Φ; the platform crate
//! maps Φ directly onto a gait mode:
//!
//!   Φ > 0.6  → Trot     (2.0 Hz, step height 0.08 m, forward 1.5 m/s)
//!   0.3–0.6  → Walk     (1.0 Hz, 0.04 m, 0.5 m/s)
//!   0.1–0.3  → Freeze   (0.0 Hz — stand in place)
//!   < 0.1    → Collapse (0.0 Hz — ragdoll)
//!
//! This is the third mode-selection demo in the C-series (after C-6
//! exoskeleton and C-8 surgical), confirming `GaitType::from_phi` as a
//! template primitive — whenever a platform crate already exposes a
//! `<Mode>::from_phi` mapping, the demo is a straight pass-through.
//!
//! - **Physics of record**: `symthaea_quadruped::SimpleQuadrupedSimulator`
//!   (CPG-driven per-leg joint dynamics, spring-damper foot contact,
//!   gait-dictated forward velocity).

mod camera;
mod consciousness_bridge;
mod controller;
mod hud;
mod plugin;
mod resources;
mod terrain;
mod visualization;

use bevy::prelude::*;
use plugin::QuadrupedDemoPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symthaea: Φ-Gated Quadruped Gait Selection (rough terrain)".into(),
                resolution: bevy::window::WindowResolution::from((1600u32, 900u32)),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(QuadrupedDemoPlugin)
        .run();
}
