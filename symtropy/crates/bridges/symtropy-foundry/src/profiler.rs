// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

pub struct PerformanceProfilerPlugin;

impl Plugin for PerformanceProfilerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Update, monitor_simulation_load);
    }
}

#[derive(Resource, Default)]
pub struct SimulationLoad {
    pub current_fps: f64,
    pub entity_count: usize,
    pub is_overloaded: bool,
}

fn monitor_simulation_load(
    diagnostics: Res<DiagnosticsStore>,
    entities: Query<Entity>,
    mut load: ResMut<SimulationLoad>,
) {
    if let Some(fps) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(value) = fps.smoothed() {
            load.current_fps = value;
            load.entity_count = entities.iter().count();

            // Throttle if FPS drops below 30
            load.is_overloaded = value < 30.0;
        }
    }
}
