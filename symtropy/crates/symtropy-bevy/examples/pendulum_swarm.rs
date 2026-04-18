// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `pendulum_swarm` — Tier 1 showcase: Phi-coupled physics on a 10x10 grid.
//!
//! See [`pendulum_swarm.md`](./pendulum_swarm.md) for the full design doc,
//! empirical constants, and step-by-step implementation plan. This file is
//! built incrementally per the plan's "Implementation steps" section.
//!
//! Current step: **1 — Hello Bevy + physics plugin.** Opens a 1280x720 window
//! with a dark background, gravity-enabled `SymtropyPhysicsPlugin::<2>`, and a
//! 2D camera. No bodies yet — the next commit adds one pendulum (step 2).

use bevy::prelude::*;
use symtropy_bevy::SymtropyPhysicsPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Symtropy: Pendulum Swarm (Phi-coupled physics)".into(),
                resolution: bevy::window::WindowResolution::from((1280u32, 720u32)),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.04, 0.04, 0.06)))
        .add_plugins(SymtropyPhysicsPlugin::<2>::with_gravity([0.0, -9.81]))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
}
