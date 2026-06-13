// Copyright (C) 2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! OLD WATERWORKS MICRO-SLICE — Ticket 1
//!
//! This is the first concrete playable foundation for Symtropy.
//! - Floor, walls, and greybox machinery (pump, tank, console).
//! - Basic first-person movement (WASD + Arrows).
//! - Proximity-based interaction with the console (Press E).
//! - Dead authority lock inspection state.
//!
//! Hard Scope:
//! - No Device Bus yet
//! - No WASM yet
//! - No Lightyear yet
//! - No Mycelix yet
//! - No Holochain yet
//! - No SymLogic yet
//! - No Field Deck UI yet
//! - No faction evolution yet
//! - No Chronicle yet
//! - just the first playable room

use bevy::prelude::*;
use symtropy_bevy_scene::{SymtropyScenePlugin, fixed_camera};

#[derive(Component)]
struct Player;

#[derive(Component)]
struct Console;

#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
enum InspectionState {
    #[default]
    None,
    PumpConsole,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Old Waterworks Micro-Slice".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(SymtropyScenePlugin::default())
        .init_state::<InspectionState>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                player_move_system.run_if(in_state(InspectionState::None)),
                interaction_system,
                inspection_ui_system,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Materials
    let floor_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.2, 0.22),
        ..default()
    });
    let wall_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.3, 0.35),
        ..default()
    });
    let pump_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.4, 0.1),
        ..default()
    });
    let tank_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.1, 0.4),
        ..default()
    });
    let console_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.1),
        ..default()
    });

    // Meshes
    let cube_mesh = meshes.add(Cuboid::default());

    // Floor
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(floor_mat),
        Transform::from_xyz(0.0, -0.1, 0.0).with_scale(Vec3::new(20.0, 0.2, 20.0)),
    ));

    // Walls
    let wall_height = 4.0;
    // North
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(wall_mat.clone()),
        Transform::from_xyz(0.0, wall_height / 2.0, -10.0).with_scale(Vec3::new(
            20.0,
            wall_height,
            0.2,
        )),
    ));
    // South
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(wall_mat.clone()),
        Transform::from_xyz(0.0, wall_height / 2.0, 10.0).with_scale(Vec3::new(
            20.0,
            wall_height,
            0.2,
        )),
    ));
    // East
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(wall_mat.clone()),
        Transform::from_xyz(10.0, wall_height / 2.0, 0.0).with_scale(Vec3::new(
            0.2,
            wall_height,
            20.0,
        )),
    ));
    // West
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(wall_mat.clone()),
        Transform::from_xyz(-10.0, wall_height / 2.0, 0.0).with_scale(Vec3::new(
            0.2,
            wall_height,
            20.0,
        )),
    ));

    // Large Pump
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(pump_mat),
        Transform::from_xyz(-2.0, 1.0, -2.0).with_scale(Vec3::new(2.0, 2.0, 2.0)),
    ));

    // Tank / Pipe
    commands.spawn((
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(tank_mat),
        Transform::from_xyz(-5.0, 1.5, 5.0).with_scale(Vec3::new(1.5, 3.0, 1.5)),
    ));

    // Console
    commands.spawn((
        Console,
        Mesh3d(cube_mesh.clone()),
        MeshMaterial3d(console_mat),
        Transform::from_xyz(8.0, 1.2, 0.0).with_scale(Vec3::new(0.5, 1.0, 1.5)),
    ));

    // Player/Camera
    commands.spawn((
        Player,
        fixed_camera(Vec3::new(0.0, 1.7, 5.0), Vec3::new(0.0, 1.7, 0.0)),
    ));

    // Light source (additional to the sun)
    commands.spawn((
        PointLight {
            intensity: 100_000.0,
            range: 20.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 3.5, 0.0),
    ));

    // UI Root for interaction hint
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont {
                    font_size: 30.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                InteractionHint,
            ));
        });

    // UI Root for Inspection
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            display: Display::None,
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.85)),
        InspectionUI,
    )).with_children(|parent| {
        parent.spawn((
            Text::new("OLD WATERWORKS CONSOLE\nPUMP_1: LOCKED\nTANK_0: 12%\nAUTHORITY: DEAD_AUTHORITY_LOCK\n\nPress Esc/Shift to drop interface"),
            TextFont {
                font_size: 40.0,
                ..default()
            },
            TextColor(Color::srgb(1.0, 0.8, 0.2)), // Amber
        ));
    });
}

#[derive(Component)]
struct InteractionHint;

#[derive(Component)]
struct InspectionUI;

fn player_move_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Player>>,
) {
    let Ok(mut transform) = query.single_mut() else {
        return;
    };
    let mut direction = Vec3::ZERO;

    if keyboard.pressed(KeyCode::KeyW) {
        let forward = transform.forward();
        direction += *forward;
    }
    if keyboard.pressed(KeyCode::KeyS) {
        let back = transform.back();
        direction += *back;
    }
    if keyboard.pressed(KeyCode::KeyA) {
        let left = transform.left();
        direction += *left;
    }
    if keyboard.pressed(KeyCode::KeyD) {
        let right = transform.right();
        direction += *right;
    }

    direction.y = 0.0;
    if direction.length_squared() > 0.0 {
        direction = direction.normalize();
        transform.translation += direction * 5.0 * time.delta_secs();
    }

    // Simple rotation with Arrows
    let mut rotation = 0.0;
    if keyboard.pressed(KeyCode::ArrowLeft) {
        rotation += 1.0;
    }
    if keyboard.pressed(KeyCode::ArrowRight) {
        rotation -= 1.0;
    }
    transform.rotate_y(rotation * 2.0 * time.delta_secs());
}

fn interaction_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    player_query: Query<&Transform, With<Player>>,
    console_query: Query<&Transform, With<Console>>,
    mut interaction_hint_query: Query<(&mut Text, &mut Visibility), With<InteractionHint>>,
    state: Res<State<InspectionState>>,
    mut next_state: ResMut<NextState<InspectionState>>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };
    let Ok(console_tf) = console_query.single() else {
        return;
    };
    let Ok((mut text, mut visibility)) = interaction_hint_query.single_mut() else {
        return;
    };

    let dist = player_tf.translation.distance(console_tf.translation);
    let near_console = dist < 2.5;

    if *state.get() == InspectionState::None {
        if near_console {
            **text = "Press E to inspect pump console".to_string();
            *visibility = Visibility::Visible;

            if keyboard.just_pressed(KeyCode::KeyE) {
                next_state.set(InspectionState::PumpConsole);
            }
        } else {
            *visibility = Visibility::Hidden;
        }
    } else {
        *visibility = Visibility::Hidden;

        if keyboard.just_pressed(KeyCode::Escape)
            || keyboard.just_pressed(KeyCode::ShiftLeft)
            || keyboard.just_pressed(KeyCode::ShiftRight)
        {
            next_state.set(InspectionState::None);
        }
    }
}

fn inspection_ui_system(
    state: Res<State<InspectionState>>,
    mut query: Query<&mut Node, With<InspectionUI>>,
) {
    let Ok(mut node) = query.single_mut() else {
        return;
    };
    if *state.get() != InspectionState::None {
        node.display = Display::Flex;
    } else {
        node.display = Display::None;
    }
}
