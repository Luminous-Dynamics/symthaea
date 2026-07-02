// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Null Ecology systems: rogue machine behavior and combat.

use crate::components::*;
use crate::resources::*;
use bevy::prelude::*;

/// System for drone AI: move toward and sabotage infrastructure.
pub fn null_drone_ai_system(
    mut drones: Query<(Entity, &mut NullDrone, &mut Transform)>,
    mut junctions: Query<
        (Entity, &mut PowerJunction, &Transform),
        (Without<NullDrone>, Without<WaterPump>),
    >,
    mut pumps: Query<
        (Entity, &mut WaterPump, &Transform),
        (Without<NullDrone>, Without<PowerJunction>),
    >,
    time: Res<Time>,
) {
    for (_drone_entity, mut drone, mut drone_tf) in &mut drones {
        // 1. Pick a target machine if we don't have one
        if drone.target_machine.is_none() {
            // Prefer damaging junctions first to cut power
            if let Some((j_ent, _, _)) = junctions.iter().find(|(_, j, _)| !j.is_damaged) {
                drone.target_machine = Some(j_ent);
            } else if let Some((p_ent, _, _)) = pumps.iter().find(|(_, p, _)| !p.is_sabotaged) {
                drone.target_machine = Some(p_ent);
            }
        }

        // 2. Move toward target
        if let Some(target_ent) = drone.target_machine {
            let target_pos = if let Ok((_, _, j_tf)) = junctions.get(target_ent) {
                Some(j_tf.translation)
            } else if let Ok((_, _, p_tf)) = pumps.get(target_ent) {
                Some(p_tf.translation)
            } else {
                drone.target_machine = None;
                None
            };

            if let Some(pos) = target_pos {
                let dir = pos - drone_tf.translation;
                let dist = dir.length();
                if dist > 10.0 {
                    let move_vec = dir.normalize() * 40.0 * time.delta_secs();
                    drone_tf.translation += move_vec;
                } else {
                    // 3. Sabotage
                    if let Ok((_, mut junction, _)) = junctions.get_mut(target_ent) {
                        junction.is_damaged = true;
                        junction.output = 0.0;
                    } else if let Ok((_, mut pump, _)) = pumps.get_mut(target_ent) {
                        pump.is_sabotaged = true;
                        pump.is_running = false;
                    }
                }
            }
        }
    }
}

/// System for player combat: click drones to destroy them.
pub fn drone_combat_system(
    mut commands: Commands,
    mouse: Res<ButtonInput<MouseButton>>,
    player: Query<&Transform, With<Player>>,
    drones: Query<(Entity, &Transform, &NullDrone)>,
    mut log: ResMut<GovernanceLog>,
    time: Res<Time>,
) {
    let Ok(player_tf) = player.single() else {
        return;
    };

    if mouse.just_pressed(MouseButton::Left) {
        for (drone_ent, drone_tf, _) in &drones {
            let dist = player_tf.translation.distance(drone_tf.translation);
            if dist < 60.0 {
                // Attack range
                commands.entity(drone_ent).despawn();
                log.push(time.elapsed_secs(), "Null Drone Neutralized".to_string(), 0);
                crate::systems::chronicle_recorder::record_chronicle_event(
                    "NullDroneNeutralized",
                    serde_json::json!({
                        "distance_to_player": dist,
                        "position": [drone_tf.translation.x, drone_tf.translation.y],
                    }),
                );
                info!("Destroyed Null Drone!");
            }
        }
    }
}

/// Spawns a wave of drones when entropy is high.
pub fn drone_spawning_system(
    mut commands: Commands,
    metrics: Res<SettlementMetrics>,
    drones: Query<&NullDrone>,
    time: Res<Time>,
    mut timer: Local<f32>,
    machines: Query<
        &Transform,
        (
            Or<(With<PowerJunction>, With<WaterPump>, With<Fabricator>)>,
            Without<NullDrone>,
        ),
    >,
    state: Res<State<GamePhase>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    *timer += time.delta_secs();
    if *timer < 10.0 {
        return;
    }
    *timer = 0.0;

    let drone_count = drones.iter().count();
    if metrics.entropy > 0.5 && drone_count < 3 {
        let is_3d = *state.get() == GamePhase::Playing3D;
        let mut spawn_pos = Vec3::new(200.0, 200.0, if is_3d { 6.0 } else { 2.5 });
        let machine_positions: Vec<Vec3> = machines.iter().map(|t| t.translation).collect();
        if !machine_positions.is_empty() {
            let idx = (time.elapsed_secs() as usize) % machine_positions.len();
            let pos = machine_positions[idx];
            let angle = time.elapsed_secs();
            let offset = Vec3::new(angle.cos() * 150.0, angle.sin() * 150.0, 0.0);
            spawn_pos = pos + offset;
            spawn_pos.z = if is_3d { 6.0 } else { 2.5 };
        }

        if is_3d {
            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(6.0).mesh().uv(16, 16))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.9, 0.1, 0.1),
                    emissive: LinearRgba::from(Color::srgb(0.5, 0.0, 0.0)),
                    perceptual_roughness: 0.3,
                    ..default()
                })),
                Transform::from_translation(spawn_pos),
                NullDrone::default(),
            ));
        } else {
            commands.spawn((
                Sprite::from_color(Color::srgb(0.9, 0.1, 0.1), Vec2::splat(12.0)),
                Transform::from_translation(spawn_pos),
                NullDrone::default(),
            ));
        }
        info!("A Null Drone has entered the basin!");
    }
}
