// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use symtropy_lightyear::SymtropyNetPlugin;
use symtropy_lightyear::components::*;
use symtropy_lightyear::protocol;

// --- Protocol ---

#[derive(Component, Serialize, Deserialize, Clone, Debug, PartialEq, Reflect)]
#[reflect(Component)]
pub struct PlayerController {
    pub id: u64,
}

// --- Demo Plugin ---

pub struct MultiplayerDemoPlugin;

impl Plugin for MultiplayerDemoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_scene);
        app.add_systems(
            Update,
            (
                handle_input,
                sync_physics_to_render,
                interest_management_debug,
            ),
        );
    }
}

fn main() {
    let mut app = App::new();

    // 1. Choose mode via CLI args
    let is_host = std::env::args().any(|a| a == "--host");
    let player_id = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: if is_host {
                format!("Symtropy Host (P{})", player_id)
            } else {
                format!("Symtropy Client (P{})", player_id)
            },
            ..default()
        }),
        ..default()
    }));

    // 2. Register Symtropy protocol
    protocol::register(&mut app);
    app.register_type::<PlayerController>();

    // 3. Add Symtropy networking plugin
    app.add_plugins(SymtropyNetPlugin);

    // 4. Add Demo logic
    app.add_plugins(MultiplayerDemoPlugin);

    app.run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Setup camera
    commands
        .spawn(Camera3d::default())
        .insert(Transform::from_xyz(0.0, 50.0, 75.0).looking_at(Vec3::ZERO, Vec3::Y));

    // Setup light
    commands.spawn(DirectionalLight {
        illuminance: 10000.0,
        shadows_enabled: true,
        ..default()
    });

    // Spawn floor
    let floor_mesh = meshes.add(Plane3d::default().mesh().size(200.0, 200.0));
    let floor_material = materials.add(Color::srgb(0.1, 0.12, 0.1));
    commands.spawn((
        Mesh3d(floor_mesh),
        MeshMaterial3d(floor_material),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Spawn placeholders for 8 players in a large village layout
    let sphere_mesh = meshes.add(Sphere::new(0.5).mesh().uv(32, 16));

    for i in 0..8 {
        let angle = (i as f32 / 8.0) * std::f32::consts::TAU;
        let radius = 30.0;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;

        let color = Color::hsl(i as f32 * 45.0, 0.9, 0.6);
        let mat = materials.add(color);

        commands.spawn((
            Mesh3d(sphere_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_xyz(x, 0.5, z),
            NetPosition {
                x: x as f64,
                y: 0.5,
                z: z as f64,
            },
            NetAuthority {
                peer_id: i as u64,
                transferable: true,
            },
            SpatialZone(0),
            PlayerController { id: i as u64 },
        ));
    }
}

fn handle_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut NetPosition, &PlayerController)>,
) {
    let mut move_vec = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        move_vec.z -= 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        move_vec.z += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        move_vec.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) {
        move_vec.x += 1.0;
    }

    if move_vec.length() > 0.0 {
        move_vec = move_vec.normalize() * 0.5;
        for (mut pos, _ctrl) in query.iter_mut() {
            pos.x += move_vec.x as f64;
            pos.z += move_vec.z as f64;
        }
    }
}

fn sync_physics_to_render(mut query: Query<(&NetPosition, &mut Transform)>) {
    for (net_pos, mut transform) in query.iter_mut() {
        transform.translation.x = net_pos.x as f32;
        transform.translation.y = net_pos.y as f32;
        transform.translation.z = net_pos.z as f32;
    }
}

fn interest_management_debug(query: Query<(&PlayerController, &SpatialZone)>) {
    for (_ctrl, _zone) in &query {
        // Log mapping if needed
    }
}
