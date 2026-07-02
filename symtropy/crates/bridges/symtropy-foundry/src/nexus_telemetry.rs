// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::orchestrator::OrchestratorObservatory;
use crate::profiler::SimulationLoad;
use bevy::prelude::*;

pub struct NexusTelemetryPlugin;

impl Plugin for NexusTelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_nexus_ui);
        app.add_systems(Update, update_nexus_ui);
    }
}

#[derive(Component)]
struct NexusUiNode;

fn setup_nexus_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                width: Val::Px(250.0),
                height: Val::Px(150.0),
                position_type: PositionType::Absolute,
                right: Val::Px(10.0),
                top: Val::Px(10.0),
                padding: UiRect::all(Val::Px(10.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.2, 0.0, 0.7)),
            NexusUiNode,
        ))
        .with_children(|parent| {
            parent.spawn(Text::new("Foundry Nexus: Ready"));
        });
}

fn update_nexus_ui(
    observatory: Res<OrchestratorObservatory>,
    load: Res<SimulationLoad>,
    mut query: Query<&mut Text, With<NexusUiNode>>,
) {
    if let Ok(mut text) = query.single_mut() {
        text.0 = format!(
            "Nexus Telemetry:\nΦ: {:.2}\nLoad: {:.2} FPS\nEntities: {}",
            observatory.avg_phi, load.current_fps, load.entity_count
        );
    }
}
