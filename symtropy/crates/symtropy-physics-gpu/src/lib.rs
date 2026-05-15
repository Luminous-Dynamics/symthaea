// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU-accelerated hierarchical spatial hash broadphase + narrowphase + integrator for Symtropy.

pub mod render;

use bevy::prelude::*;
use bevy::render::{
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, RenderSubGraph},
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderQueue},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    storage::{ShaderStorageBuffer, GpuShaderStorageBuffer},
    RenderApp, Render, RenderSystems,
};
use bevy::asset::RenderAssetUsages;
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use symtropy_bevy_core::PhysicsBody;

pub use render::{InstancedPhysicsMaterial, GpuInstancedMesh};

pub mod shape {
    pub const SPHERE: u32 = 0;
    pub const CUBOID: u32 = 1;
}

/// Generic GPU-compatible collider data (Static/Kinematic properties).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct GpuCollider {
    pub translation: [f32; 3],
    pub _pad1: f32, 
    pub rotation: [f32; 4],
    pub half_extents: [f32; 3],
    pub shape_type: u32,
    pub body_index: u32,
    pub _pad2: u32, 
    pub _pad3: u32,
    pub _pad4: u32,
}

/// Dynamic physics state for integration and solving.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct GpuPhysicsState {
    pub velocity: [f32; 3],
    pub inv_mass: f32,
    pub angular_velocity: [f32; 3],
    pub friction: f32,
}

/// Instance data for high-performance rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct GpuInstanceData {
    pub model_matrix: Mat4,
}

/// Potential collision pair found by the GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct GpuCollisionPair {
    pub body_a: u32,
    pub body_b: u32,
}

/// Uniform configuration for the physics pipeline.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType, Default)]
pub struct BroadphaseConfig {
    pub cell_size: f32,
    pub grid_dim: u32,
    pub max_pairs: u32,
    pub num_bodies: u32,
    pub dt: f32,
}

/// Resource managing GPU physics state and buffers.
#[derive(Resource, ExtractResource, Clone, Default)]
pub struct GpuBroadphaseManager {
    pub config: BroadphaseConfig,
    pub colliders: Vec<GpuCollider>,
    pub physics_states: Vec<GpuPhysicsState>,
    pub instance_buffer: Handle<ShaderStorageBuffer>,
}

#[derive(Resource, Default)]
pub struct BroadphaseResults {
    pub pair_count: u32,
    pub pairs: Vec<GpuCollisionPair>,
    pub hero_map: std::collections::HashMap<u32, Entity>,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderSubGraph)]
pub struct BroadphaseSubGraph;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct BroadphaseLabel;

pub struct GpuPhysicsPlugin;

impl Plugin for GpuPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<InstancedPhysicsMaterial>::default())
            .add_plugins(ExtractComponentPlugin::<GpuInstancedMesh>::default())
            .init_resource::<GpuBroadphaseManager>()
            .init_resource::<BroadphaseResults>()
            .add_plugins(ExtractResourcePlugin::<GpuBroadphaseManager>::default())
            .add_systems(Update, (upload_physics_to_gpu_sparse, readback_results, debug_draw_broadphase));

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<BroadphasePipeline>()
                .add_systems(Render, prepare_buffers)
                .add_systems(Render, override_instance_counts.in_set(RenderSystems::Queue))
                .add_render_graph_node::<BroadphaseComputeNode>(
                    BroadphaseSubGraph,
                    BroadphaseLabel,
                );
        }
    }
}

impl ExtractComponent for GpuInstancedMesh {
    type QueryData = &'static GpuInstancedMesh;
    type QueryFilter = ();
    type Out = GpuInstancedMesh;

    fn extract_component(item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(GpuInstancedMesh { instance_count: item.instance_count })
    }
}

/// Sparse Sync: Only mirror Hero NPCs to ECS, the rest stay on GPU.
fn upload_physics_to_gpu_sparse(
    mut manager: ResMut<GpuBroadphaseManager>,
    mut results: ResMut<BroadphaseResults>,
    bodies: Query<(Entity, &PhysicsBody, &GlobalTransform)>,
    camera: Query<&GlobalTransform, With<Camera3d>>,
    time: Res<Time>,
    mut storage_buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    let camera_pos = camera.iter().next().map(|c| c.translation()).unwrap_or(Vec3::ZERO);
    const HERO_RADIUS: f32 = 100.0; 

    manager.colliders.clear();
    manager.physics_states.clear();
    results.hero_map.clear();
    manager.config.dt = time.delta_secs();

    for (entity, _body, transform) in bodies.iter() {
        let (scale, rotation, translation) = transform.to_scale_rotation_translation();
        let idx = manager.colliders.len() as u32;

        if translation.distance(camera_pos) < HERO_RADIUS {
            results.hero_map.insert(idx, entity);
        }

        manager.colliders.push(GpuCollider {
            translation: translation.into(),
            _pad1: 0.0,
            rotation: rotation.into(),
            half_extents: scale.into(), 
            shape_type: shape::SPHERE,
            body_index: idx,
            _pad2: 0, _pad3: 0, _pad4: 0,
        });

        manager.physics_states.push(GpuPhysicsState {
            velocity: [0.0; 3], 
            inv_mass: 1.0,
            angular_velocity: [0.0; 3],
            friction: 0.5,
        });
    }

    manager.config.num_bodies = manager.colliders.len() as u32;
    manager.config.max_pairs = (manager.config.num_bodies * 8).max(100_000);

    if storage_buffers.get(&manager.instance_buffer).is_none() {
        let buffer = vec![GpuInstanceData { model_matrix: Mat4::IDENTITY }; 300_000];
        manager.instance_buffer = storage_buffers.add(ShaderStorageBuffer::new(bytemuck::cast_slice(&buffer), RenderAssetUsages::default()));
    }
}

fn override_instance_counts(
    mut _query: Query<(&GpuInstancedMesh, &mut ViewVisibility)>,
) {
    // Placeholder
}

#[derive(Resource)]
struct BroadphasePipeline {
    count_scatter_pipeline: CachedComputePipelineId,
    broadphase_pipeline: CachedComputePipelineId,
    integrate_pipeline: CachedComputePipelineId,
    bind_group_layout: BindGroupLayout,
}

impl FromWorld for BroadphasePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let shader = asset_server.load("shaders/spatial_hash_broadphase.wgsl");

        let entries = vec![
            BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            BindGroupLayoutEntry { binding: 6, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            BindGroupLayoutEntry { binding: 7, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            BindGroupLayoutEntry { binding: 8, visibility: ShaderStages::COMPUTE | ShaderStages::VERTEX, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ];

        let bind_group_layout = render_device.create_bind_group_layout(
            Some("gpu_physics_bind_group_layout"),
            &entries,
        );

        let count_scatter_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("Broadphase Count and Scatter")),
            layout: vec![BindGroupLayoutDescriptor { label: Cow::from("Layout"), entries: entries.clone() }],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("count_and_scatter")),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let broadphase_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("Broadphase + Narrowphase")),
            layout: vec![BindGroupLayoutDescriptor { label: Cow::from("Layout"), entries: entries.clone() }],
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("broadphase")),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        let integrate_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("Physics Integration")),
            layout: vec![BindGroupLayoutDescriptor { label: Cow::from("Layout"), entries }],
            shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("integrate")),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        });

        Self {
            count_scatter_pipeline,
            broadphase_pipeline,
            integrate_pipeline,
            bind_group_layout,
        }
    }
}

#[derive(Resource)]
struct BroadphaseBuffers {
    bind_group: BindGroup,
    num_bodies: u32,
}

fn prepare_buffers(
    manager: Res<GpuBroadphaseManager>,
    render_device: Res<RenderDevice>,
    pipeline: Res<BroadphasePipeline>,
    mut commands: Commands,
) {
    if manager.colliders.is_empty() {
        return;
    }

    let colliders_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Colliders"),
        contents: bytemuck::cast_slice(&manager.colliders),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let output_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Output Pairs"),
        size: (manager.config.max_pairs as u64) * std::mem::size_of::<GpuCollisionPair>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let count_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Pair Count"),
        contents: bytemuck::bytes_of(&0u32),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    let config_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Config"),
        contents: bytemuck::bytes_of(&manager.config),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let coarse_dim = (manager.config.grid_dim / 8).max(1);
    let coarse_mask_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Coarse Mask"),
        contents: bytemuck::cast_slice(&vec![0u32; (coarse_dim as usize).pow(3)]),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let cell_head_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Cell Head"),
        contents: bytemuck::cast_slice(&vec![-1i32; (manager.config.grid_dim as usize).pow(3)]),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let node_next_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Node Next"),
        size: (manager.config.num_bodies * 4) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let physics_state_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Physics State"),
        contents: bytemuck::cast_slice(&manager.physics_states),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    let instance_data_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Instance Data"),
        size: (manager.config.num_bodies as u64) * std::mem::size_of::<GpuInstanceData>() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = render_device.create_bind_group(
        Some("gpu_physics_bind_group"),
        &pipeline.bind_group_layout,
        &[
            BindGroupEntry { binding: 0, resource: colliders_buffer.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: count_buffer.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: config_buffer.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: coarse_mask_buffer.as_entire_binding() },
            BindGroupEntry { binding: 5, resource: cell_head_buffer.as_entire_binding() },
            BindGroupEntry { binding: 6, resource: node_next_buffer.as_entire_binding() },
            BindGroupEntry { binding: 7, resource: physics_state_buffer.as_entire_binding() },
            BindGroupEntry { binding: 8, resource: instance_data_buffer.as_entire_binding() },
        ],
    );

    commands.insert_resource(BroadphaseBuffers {
        bind_group,
        num_bodies: manager.config.num_bodies,
    });
}

#[derive(Default)]
pub struct BroadphaseComputeNode;

impl Node for BroadphaseComputeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<BroadphasePipeline>();
        let buffers = match world.get_resource::<BroadphaseBuffers>() {
            Some(b) => b,
            None => return Ok(()),
        };

        let count_p = pipeline_cache.get_compute_pipeline(pipeline.count_scatter_pipeline).unwrap();
        let broadphase_p = pipeline_cache.get_compute_pipeline(pipeline.broadphase_pipeline).unwrap();
        let integrate_p = pipeline_cache.get_compute_pipeline(pipeline.integrate_pipeline).unwrap();

        let mut pass = render_context.command_encoder().begin_compute_pass(&ComputePassDescriptor {
            label: Some("GPU Physics Pipeline"),
            timestamp_writes: None,
        });

        let workgroups = (buffers.num_bodies + 127) / 128;

        pass.set_bind_group(0, &buffers.bind_group, &[]);
        
        pass.set_pipeline(count_p);
        pass.dispatch_workgroups(workgroups, 1, 1);

        pass.set_pipeline(broadphase_p);
        pass.dispatch_workgroups(workgroups, 1, 1);

        pass.set_pipeline(integrate_p);
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }
}

/// Readback results from GPU to CPU (Hero NPCs only).
fn readback_results(
    _render_device: Res<RenderDevice>,
    _render_queue: Res<RenderQueue>,
    _buffers: Option<Res<BroadphaseBuffers>>,
    mut results: ResMut<BroadphaseResults>,
) {
    results.pair_count = 0;
    results.pairs.clear();
}

/// Debug visualization.
fn debug_draw_broadphase(
    mut gizmos: Gizmos,
    manager: Res<GpuBroadphaseManager>,
    _results: Res<BroadphaseResults>,
) {
    for collider in &manager.colliders {
        let size = Vec3::from(collider.half_extents) * 2.0;
        gizmos.primitive_3d(&Cuboid::from_size(size), Vec3::from(collider.translation), Color::hsla(0.0, 1.0, 0.5, 0.1));
    }
}
