// SPDX-License-Identifier: Apache-2.0 OR MIT

use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
pub struct InstancedPhysicsMaterial {
    #[uniform(0)]
    pub color: LinearRgba,
}

impl Material for InstancedPhysicsMaterial {}

#[derive(Component)]
pub struct GpuInstancedMesh;
