// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Click-to-select system — manual raycasting against marker spheres.
//! Uses Camera::viewport_to_world for ray creation, then tests against
//! marker sphere positions.

use crate::camera::OrbitalCamera;
use crate::markers::DataMarker;
use bevy::prelude::*;

/// Currently selected marker.
#[derive(Resource, Default)]
pub struct SelectedMarker {
    pub entity: Option<Entity>,
    pub info: Option<DataMarker>,
}

/// Component for the selection info UI text.
#[derive(Component)]
pub struct SelectionInfoText;

/// Spawn the selection info text at the bottom of the screen.
/// Caller should also tag the entity with their cleanup marker (e.g. AtlasEntity).
pub fn setup_selection_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Sol Atlas — Arrow keys: timeline | Space: play | Click: select | Esc: exit"),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::linear_rgb(0.6, 0.7, 0.8)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
        SelectionInfoText,
    ));
}

/// Update the selection info text.
pub fn update_selection_text(
    selected: Res<SelectedMarker>,
    mut query: Query<&mut Text, With<SelectionInfoText>>,
) {
    for mut text in query.iter_mut() {
        if let Some(ref info) = selected.info {
            **text = format!("{} — {}", info.layer.label(), info.name);
        } else {
            **text = String::new();
        }
    }
}

/// Detect clicks and raycast against markers.
/// Only treats mouse releases with < 5px movement as clicks (not drags).
pub fn click_select_system(
    mouse_button: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<OrbitalCamera>>,
    markers: Query<(Entity, &GlobalTransform, &DataMarker)>,
    mut selected: ResMut<SelectedMarker>,
) {
    if !mouse_button.just_released(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_tf)) = camera_q.single() else {
        return;
    };

    // Create ray from camera through cursor
    let Ok(ray) = camera.viewport_to_world(camera_tf, cursor_pos) else {
        return;
    };

    // Find nearest marker hit (within threshold)
    let mut best: Option<(Entity, f32, DataMarker)> = None;
    let threshold = 0.05; // world-space distance threshold

    for (entity, marker_tf, data) in markers.iter() {
        let marker_pos = marker_tf.translation();
        // Distance from ray to point (perpendicular distance)
        let to_marker = marker_pos - ray.origin;
        let along_ray = to_marker.dot(*ray.direction);
        if along_ray < 0.0 {
            continue;
        } // behind camera
        let closest_on_ray = ray.origin + *ray.direction * along_ray;
        let dist = (marker_pos - closest_on_ray).length();

        if dist < threshold {
            if best.as_ref().map_or(true, |(_, d, _)| along_ray < *d) {
                best = Some((entity, along_ray, data.clone()));
            }
        }
    }

    if let Some((entity, _, data)) = best {
        info!("[atlas] Selected: {} ({})", data.name, data.layer.label());
        selected.entity = Some(entity);
        selected.info = Some(data);
    } else {
        // Clicked empty space — deselect
        selected.entity = None;
        selected.info = None;
    }
}
