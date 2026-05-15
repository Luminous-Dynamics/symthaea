// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Symtropy 4D XR Showcase.
//! Renders the 4D pendulum swarm within a spatial XR context.

use bevy::prelude::*;
use bevy_openxr::prelude::*;
use symtropy_bevy::plugin::SymtropyPhysicsPlugin;
use symtropy_render_bridge::material::NdSlicingPlugin;

fn main() {
    App::new()
        // 1. Initialize OpenXR backend
        .add_plugins(OpenXrPlugin::default())
        .add_plugins(DefaultXrPlugins)
        // 2. Load the Symtropy Engine (D=4)
        .add_plugins(SymtropyPhysicsPlugin::<4>::with_gravity([
            0.0, -9.81, 0.0, 0.0,
        ]))
        .add_plugins(NdSlicingPlugin)
        // 3. Setup the 4D visualization space
        .add_systems(Startup, setup_xr_space)
        .run();
}

fn setup_xr_space(mut _commands: Commands) {
    // Note: The OpenXrPlugin handles spawning the spatial camera rig.
    // We can add spatial UI or 4D anchors here.
    bevy_log::info!("Symtropy 4D XR Showcase Initialized.");
}
