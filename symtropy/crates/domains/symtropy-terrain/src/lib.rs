// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Volumetric terrain system for Symtropy, supporting subterranean voids,
//! chemical weathering, and permeability-driven seepage.

use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;
use rapier3d::prelude::*;
use symtropy_rapier3d_bridge::{
    RapierColliderSet, RapierImpulseJointSet, RapierIslandManager, RapierMultibodyJointSet,
    RapierRigidBodySet,
};

use symtropy_physics_gpu::HybridFluidPlugin;

pub const CHUNK_SIZE: usize = 16;

/// The material properties of the subterranean earth.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Reflect)]
pub enum SubstrateMaterial {
    #[default]
    Air,
    Bedrock,
    Dolomite,
    PyriteTailing,
    Quartzite,
}

/// A 3D volumetric chunk of earth that can be excavated or weathered.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct EarthChunk {
    pub voxels: [[[SubstrateMaterial; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    pub densities: [[[f32; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    pub is_dirty: bool,
    pub is_rebuilding: bool,
    #[reflect(ignore)]
    pub rigid_body: Option<RigidBodyHandle>,
    #[reflect(ignore)]
    pub mesh_handle: Handle<Mesh>,
}

impl Default for EarthChunk {
    fn default() -> Self {
        Self {
            voxels: [[[SubstrateMaterial::Dolomite; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
            densities: [[[1.0; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
            is_dirty: true,
            is_rebuilding: false,
            rigid_body: None,
            mesh_handle: Default::default(),
        }
    }
}

pub struct RebuildResult {
    pub entity: Entity,
    pub rigid_body: RigidBody,
    pub colliders: Vec<Collider>,
    pub mesh: Mesh,
}

#[derive(Component)]
pub struct RebuildTask(pub Task<RebuildResult>);

#[derive(Message, Debug, Clone)]
pub struct WeatheringEvent {
    pub chunk_entity: Entity,
    pub voxel_idx: u32,
}

#[derive(Message, Debug, Clone)]
pub struct ExcavationEvent {
    pub world_position: Vec3,
    pub radius: f32,
}

pub struct SymtropyTerrainPlugin;

impl Plugin for SymtropyTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(HybridFluidPlugin)
            .register_type::<EarthChunk>()
            .add_message::<ExcavationEvent>()
            .add_message::<WeatheringEvent>()
            .add_systems(
                Update,
                (
                    process_excavation,
                    spawn_rebuild_tasks,
                    handle_rebuild_tasks,
                ),
            );
    }
}

pub fn process_excavation(
    mut messages: MessageReader<ExcavationEvent>,
    mut chunk_query: Query<(&GlobalTransform, &mut EarthChunk)>,
) {
    for event in messages.read() {
        for (chunk_tf, mut chunk) in &mut chunk_query {
            let local_p = event.world_position - chunk_tf.translation();
            if local_p.length() > event.radius + (CHUNK_SIZE as f32 * 1.7) {
                continue;
            }

            let mut modified = false;
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    for z in 0..CHUNK_SIZE {
                        let voxel_pos = Vec3::new(x as f32, y as f32, z as f32);
                        if voxel_pos.distance(local_p) <= event.radius {
                            if chunk.voxels[x][y][z] != SubstrateMaterial::Air
                                && chunk.voxels[x][y][z] != SubstrateMaterial::Bedrock
                            {
                                chunk.voxels[x][y][z] = SubstrateMaterial::Air;
                                chunk.densities[x][y][z] = 0.0;
                                modified = true;
                            }
                        }
                    }
                }
            }
            if modified {
                chunk.is_dirty = true;
            }
        }
    }
}

pub fn spawn_rebuild_tasks(
    mut commands: Commands,
    mut chunk_query: Query<(Entity, &GlobalTransform, &mut EarthChunk)>,
) {
    let pool = AsyncComputeTaskPool::get();

    for (entity, chunk_tf, mut chunk) in &mut chunk_query {
        if chunk.is_dirty && !chunk.is_rebuilding {
            let voxels = chunk.voxels;
            let translation = chunk_tf.translation();
            chunk.is_rebuilding = true;
            chunk.is_dirty = false;

            let task = pool.spawn(async move {
                let mut colliders = Vec::new();
                let mut positions = Vec::new();
                let mut normals = Vec::new();
                let mut indices = Vec::new();

                for x in 0..CHUNK_SIZE {
                    for y in 0..CHUNK_SIZE {
                        for z in 0..CHUNK_SIZE {
                            if voxels[x][y][z] == SubstrateMaterial::Air {
                                continue;
                            }
                            let voxel_p = Vec3::new(x as f32, y as f32, z as f32);
                            colliders.push(
                                ColliderBuilder::cuboid(0.5, 0.5, 0.5)
                                    .translation(vector![voxel_p.x, voxel_p.y, voxel_p.z])
                                    .build(),
                            );

                            add_voxel_mesh(
                                x,
                                y,
                                z,
                                &voxels,
                                &mut positions,
                                &mut normals,
                                &mut indices,
                            );
                        }
                    }
                }

                let mut mesh = Mesh::new(
                    bevy::render::render_resource::PrimitiveTopology::TriangleList,
                    bevy::asset::RenderAssetUsages::default(),
                );
                mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
                mesh.insert_indices(bevy_mesh::Indices::U32(indices));

                let rb = RigidBodyBuilder::fixed()
                    .translation(vector![translation.x, translation.y, translation.z])
                    .build();

                RebuildResult {
                    entity,
                    rigid_body: rb,
                    colliders,
                    mesh,
                }
            });

            commands.entity(entity).insert(RebuildTask(task));
        }
    }
}

fn add_voxel_mesh(
    x: usize,
    y: usize,
    z: usize,
    voxels: &[[[SubstrateMaterial; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    pos: &mut Vec<[f32; 3]>,
    norm: &mut Vec<[f32; 3]>,
    idx: &mut Vec<u32>,
) {
    let p = [x as f32, y as f32, z as f32];
    let dirs = [
        ([1.0, 0.0, 0.0], [1, 0, 0]),
        ([-1.0, 0.0, 0.0], [-1, 0, 0]),
        ([0.0, 1.0, 0.0], [0, 1, 0]),
        ([0.0, -1.0, 0.0], [0, -1, 0]),
        ([0.0, 0.0, 1.0], [0, 0, 1]),
        ([0.0, 0.0, -1.0], [0, 0, -1]),
    ];

    for (n_vec, offset) in dirs {
        let nx = x as i32 + offset[0];
        let ny = y as i32 + offset[1];
        let nz = z as i32 + offset[2];

        let is_exposed = nx < 0
            || nx >= CHUNK_SIZE as i32
            || ny < 0
            || ny >= CHUNK_SIZE as i32
            || nz < 0
            || nz >= CHUNK_SIZE as i32
            || voxels[nx as usize][ny as usize][nz as usize] == SubstrateMaterial::Air;

        if is_exposed {
            let start_idx = pos.len() as u32;
            for _ in 0..4 {
                norm.push(n_vec);
            }
            pos.push([p[0] - 0.5, p[1] - 0.5, p[2] - 0.5]);
            pos.push([p[0] + 0.5, p[1] - 0.5, p[2] - 0.5]);
            pos.push([p[0] + 0.5, p[1] + 0.5, p[2] - 0.5]);
            pos.push([p[0] - 0.5, p[1] + 0.5, p[2] - 0.5]);
            idx.extend_from_slice(&[
                start_idx,
                start_idx + 1,
                start_idx + 2,
                start_idx,
                start_idx + 2,
                start_idx + 3,
            ]);
        }
    }
}

pub fn handle_rebuild_tasks(
    mut commands: Commands,
    mut chunk_query: Query<&mut EarthChunk>,
    mut tasks_query: Query<(Entity, &mut RebuildTask)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut rigid_body_set_res: ResMut<RapierRigidBodySet>,
    mut collider_set_res: ResMut<RapierColliderSet>,
    mut impulse_joint_set_res: ResMut<RapierImpulseJointSet>,
    mut island_manager_res: ResMut<RapierIslandManager>,
    mut multibody_joint_set_res: ResMut<RapierMultibodyJointSet>,
) {
    let rigid_body_set = &mut rigid_body_set_res.0;
    let collider_set = &mut collider_set_res.0;
    let impulse_joint_set = &mut impulse_joint_set_res.0;
    let island_manager = &mut island_manager_res.0;
    let multibody_joint_set = &mut multibody_joint_set_res.0;

    for (task_entity, mut task) in &mut tasks_query {
        if let Some(result) = future::block_on(future::poll_once(&mut task.0)) {
            if let Ok(mut chunk) = chunk_query.get_mut(result.entity) {
                if let Some(handle) = chunk.rigid_body {
                    rigid_body_set.remove(
                        handle,
                        island_manager,
                        collider_set,
                        impulse_joint_set,
                        multibody_joint_set,
                        true,
                    );
                }
                let rb_handle = rigid_body_set.insert(result.rigid_body);
                for c in result.colliders {
                    collider_set.insert_with_parent(c, rb_handle, rigid_body_set);
                }
                chunk.rigid_body = Some(rb_handle);

                let mesh_h = meshes.add(result.mesh);
                chunk.mesh_handle = mesh_h.clone();
                commands.entity(result.entity).insert(Mesh3d(mesh_h));
                chunk.is_rebuilding = false;
            }
            commands.entity(task_entity).remove::<RebuildTask>();
        }
    }
}
