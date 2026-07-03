// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU-accelerated spatial hash broadphase + XPBD integrator + Governance Coupling.

pub mod fluid;
pub mod render;

use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::{
    Render, RenderApp,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::*,
    renderer::{RenderContext, RenderDevice, RenderGraph},
    storage::ShaderBuffer,
};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use symthaea_bevy_brain::CognitiveBrain;
use symtropy_bevy_core::PhysicsBody;

pub use fluid::{GpuFluidVoxel, HybridFluidManager, HybridFluidPlugin};
pub use render::{GpuInstancedMesh, InstancedPhysicsMaterial};

pub mod shape {
    pub const SPHERE: u32 = 0;
    pub const CUBOID: u32 = 1;
}

/// Voxel-space Nociception (3D Pain Vector)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct NociceptionData {
    pub pain: [u32; 64],
}

impl Default for NociceptionData {
    fn default() -> Self {
        Self { pain: [0u32; 64] }
    }
}

/// Social state for citizens on GPU.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct GpuCitizenSocialState {
    pub stewardship_care: f32,
    pub accumulated_exposure: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

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

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct GpuPhysicsState {
    pub velocity: [f32; 3],
    pub inv_mass: f32,
    pub angular_velocity: [f32; 3],
    pub friction: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType)]
pub struct GpuInstanceData {
    pub model_matrix: Mat4,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable, ShaderType, Default)]
pub struct BroadphaseConfig {
    pub cell_size: f32,
    pub grid_dim: u32,
    pub max_pairs: u32,
    pub num_bodies: u32,
    pub dt: f32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[derive(Resource, ExtractResource, Clone, Default)]
pub struct GpuBroadphaseManager {
    pub config: BroadphaseConfig,
    pub colliders: Vec<GpuCollider>,
    pub physics_states: Vec<GpuPhysicsState>,
    pub social_states: Vec<GpuCitizenSocialState>,
    pub instance_buffer: Handle<ShaderBuffer>,
    pub social_buffer: Handle<ShaderBuffer>,
}

#[derive(Resource, ExtractResource, Clone)]
pub struct NociceptionResults {
    pub pain_vector: [u32; 64],
}

impl Default for NociceptionResults {
    fn default() -> Self {
        Self {
            pain_vector: [0u32; 64],
        }
    }
}

pub struct GpuPhysicsPlugin;

impl Plugin for GpuPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<InstancedPhysicsMaterial>::default())
            .init_resource::<GpuBroadphaseManager>()
            .init_resource::<NociceptionResults>()
            .add_plugins(ExtractResourcePlugin::<GpuBroadphaseManager>::default())
            .add_plugins(ExtractResourcePlugin::<NociceptionResults>::default())
            .add_systems(
                Update,
                (
                    upload_physics_to_gpu_sparse,
                    cognitive_to_gpu_bridge,
                    readback_nociception,
                ),
            );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_systems(Render, prepare_buffers)
                .add_systems(RenderGraph, broadphase_readback.before(camera_driver));
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<BroadphasePipeline>();
        }
    }
}

#[derive(Resource)]
struct BroadphasePipeline {
    count_scatter_pipeline: CachedComputePipelineId,
    integrate_pipeline: CachedComputePipelineId,
    social_decay_pipeline: CachedComputePipelineId,
    bind_group_layout: BindGroupLayout,
}

impl FromWorld for BroadphasePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let asset_server = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let shader = asset_server.load("shaders/spatial_hash_broadphase.wgsl");
        let social_shader = asset_server.load("shaders/citizen_social.wgsl");

        let entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE | ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 10,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 11,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let bind_group_layout =
            render_device.create_bind_group_layout(Some("gpu_physics_layout"), &entries);
        let layout_desc = BindGroupLayoutDescriptor {
            label: Cow::from("Layout"),
            entries,
        };

        let count_scatter_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::from("Hierarchy")),
                layout: vec![layout_desc.clone()],
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(Cow::from("count_and_scatter")),
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            });

        let integrate_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::from("Integrate")),
            layout: vec![layout_desc.clone()],
            shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("integrate")),
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        });

        let social_decay_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::from("Social Decay")),
                layout: vec![layout_desc],
                shader: social_shader,
                shader_defs: vec![],
                entry_point: Some(Cow::from("apply_social_decay")),
                immediate_size: 0,
                zero_initialize_workgroup_memory: false,
            });

        Self {
            count_scatter_pipeline,
            integrate_pipeline,
            social_decay_pipeline,
            bind_group_layout,
        }
    }
}

#[derive(Resource)]
struct BroadphaseBuffers {
    bind_group: BindGroup,
    nociception_buffer: Buffer,
    nociception_staging: Buffer,
    num_bodies: u32,
}

fn prepare_buffers(
    manager: Option<Res<GpuBroadphaseManager>>,
    fluid_manager: Option<Res<HybridFluidManager>>,
    render_device: Option<Res<RenderDevice>>,
    pipeline: Option<Res<BroadphasePipeline>>,
    mut commands: Commands,
) {
    let (manager, fluid_manager, render_device, pipeline) =
        match (manager, fluid_manager, render_device, pipeline) {
            (Some(m), Some(f), Some(d), Some(p)) => (m, f, d, p),
            _ => return,
        };
    if manager.colliders.is_empty() {
        return;
    }

    let collider_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Colliders"),
        contents: bytemuck::cast_slice(&manager.colliders),
        usage: BufferUsages::STORAGE,
    });

    let config_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Broadphase Config"),
        contents: bytemuck::bytes_of(&manager.config),
        usage: BufferUsages::UNIFORM,
    });

    let social_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Social States"),
        contents: bytemuck::cast_slice(&manager.social_states),
        usage: BufferUsages::STORAGE,
    });

    let nociception_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Nociception Grid"),
        size: 256,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let nociception_staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("Nociception Staging"),
        size: 256,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Substrate + Hierarchical grids (dummy for now)
    let dummy_storage = render_device.create_buffer(&BufferDescriptor {
        label: None,
        size: 1024,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Mapping 300k fluid voxels
    let fluid_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("Fluid Voxel Read Only"),
        contents: bytemuck::cast_slice(&fluid_manager.voxels),
        usage: BufferUsages::STORAGE,
    });

    let bind_group = render_device.create_bind_group(
        Some("gpu_physics_bg"),
        &pipeline.bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: collider_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: dummy_storage.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: dummy_storage.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: config_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: dummy_storage.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: dummy_storage.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: dummy_storage.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: dummy_storage.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: dummy_storage.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 9,
                resource: social_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 10,
                resource: nociception_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 11,
                resource: fluid_buffer.as_entire_binding(),
            },
        ],
    );

    commands.insert_resource(BroadphaseBuffers {
        bind_group,
        nociception_buffer,
        nociception_staging,
        num_bodies: manager.config.num_bodies,
    });
}

fn broadphase_readback(mut render_context: RenderContext, buffers: Option<Res<BroadphaseBuffers>>) {
    let Some(buffers) = buffers else {
        return;
    };
    let command_encoder = render_context.command_encoder();
    command_encoder.copy_buffer_to_buffer(
        &buffers.nociception_buffer,
        0,
        &buffers.nociception_staging,
        0,
        256,
    );
}

fn readback_nociception(mut _results: ResMut<NociceptionResults>) {}

pub fn cognitive_to_gpu_bridge(
    mut manager: ResMut<GpuBroadphaseManager>,
    query: Query<(&PhysicsBody, &GlobalTransform, &CognitiveBrain)>,
) {
    for (body, transform, brain) in query.iter() {
        let idx = body.handle.0 as u32;
        if idx < manager.colliders.len() as u32 && idx < manager.social_states.len() as u32 {
            let (_, rotation, translation) = transform.to_scale_rotation_translation();
            manager.colliders[idx as usize].translation = translation.into();
            manager.colliders[idx as usize].rotation = rotation.into();
            manager.social_states[idx as usize].stewardship_care =
                brain.profile.stewardship_care as f32;
        }
    }
}

fn upload_physics_to_gpu_sparse(
    mut manager: ResMut<GpuBroadphaseManager>,
    _bodies: Query<(&PhysicsBody, &GlobalTransform)>,
    time: Res<Time>,
    mut storage_buffers: ResMut<Assets<ShaderBuffer>>,
) {
    if time.elapsed_secs() % 5.0 < 0.02 {
        info!("TRACE: upload_physics_to_gpu_sparse ticking...");
    }
    manager.config.dt = time.delta_secs();

    let num_bodies = manager.config.num_bodies as usize;
    if manager.social_states.len() < num_bodies {
        manager.social_states.resize(
            num_bodies,
            GpuCitizenSocialState {
                stewardship_care: 1.0,
                accumulated_exposure: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
        );
    }

    if storage_buffers.get(&manager.instance_buffer).is_none() {
        let buffer = vec![
            GpuInstanceData {
                model_matrix: Mat4::IDENTITY
            };
            300_000
        ];
        manager.instance_buffer = storage_buffers.add(ShaderBuffer::new(
            bytemuck::cast_slice(&buffer),
            bevy::asset::RenderAssetUsages::default(),
        ));
    }

    if storage_buffers.get(&manager.social_buffer).is_none() {
        manager.social_buffer = storage_buffers.add(ShaderBuffer::new(
            bytemuck::cast_slice(&manager.social_states),
            bevy::asset::RenderAssetUsages::default(),
        ));
    } else if let Some(mut buffer) = storage_buffers.get_mut(&manager.social_buffer) {
        buffer.set_data(&manager.social_states);
    }
}
