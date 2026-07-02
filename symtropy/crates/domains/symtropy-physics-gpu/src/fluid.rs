// SPDX-License-Identifier: Apache-2.0 OR MIT

use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

pub struct HybridFluidPlugin;

impl Plugin for HybridFluidPlugin {
    fn build(&self, _app: &mut App) {}
}

#[derive(Resource, Default)]
pub struct HybridFluidManager {
    pub voxels: Vec<GpuFluidVoxel>,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
pub struct GpuFluidVoxel {
    pub position: [f32; 3],
    pub density: f32,
}
