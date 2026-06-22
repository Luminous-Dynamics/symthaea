// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Automated persistence for the crystallized spacetime field.

use crate::celestial::CrystalFieldResource;
use bevy::prelude::*;

/// Plugin that handles automated load/save cycles for the CrystalField.
pub struct CrystallizedRegistryPlugin;

impl Plugin for CrystallizedRegistryPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, load_crystal_system)
            .add_systems(Last, save_crystal_system);
    }
}

fn load_crystal_system(mut res: ResMut<CrystalFieldResource>) {
    res.load();
}

fn save_crystal_system(res: Res<CrystalFieldResource>) {
    res.save();
}
