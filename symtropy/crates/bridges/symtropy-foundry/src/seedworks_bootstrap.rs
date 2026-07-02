// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! `symtropy-foundry` — Infrastructure-specific spawning rules for Seedworks.

use crate::FoundryAsset;
use bevy::prelude::*;

/// Plugin that handles Seedworks-specific infrastructure spawning.
pub struct SeedworksBootstrapPlugin;

impl Plugin for SeedworksBootstrapPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, bootstrap_firstlight_basin);
    }
}

/// System to instantiate the core critical infrastructure for Firstlight Basin.
fn bootstrap_firstlight_basin(mut commands: Commands) {
    info!("Foundry: Bootstrapping Firstlight Basin infrastructure...");

    // Water Pump
    commands.spawn((
        Name::new("WaterPump_01"),
        Transform::default(),
        Visibility::default(),
        FoundryAsset {
            id: "seedworks.infra.water_pump.001".to_string(),
        },
    ));

    // Power Junction
    commands.spawn((
        Name::new("PowerJunction_01"),
        Transform::default(),
        Visibility::default(),
        FoundryAsset {
            id: "seedworks.infra.power_junction.001".to_string(),
        },
    ));
}
