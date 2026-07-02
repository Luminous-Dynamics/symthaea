// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

pub struct SnapshotPlugin;

impl Plugin for SnapshotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, handle_snapshot_triggers);
    }
}

#[derive(Serialize, Deserialize, Resource)]
pub struct SnapshotMetadata {
    pub epoch: u32,
    pub world_name: String,
}

fn handle_snapshot_triggers(_world: &World, input: Res<ButtonInput<KeyCode>>) {
    if input.just_pressed(KeyCode::F5) {
        info!("Foundry: Capturing Snapshot...");
        // In Bevy 0.18, we use `bevy_reflect` to serialize components.
    }
}
