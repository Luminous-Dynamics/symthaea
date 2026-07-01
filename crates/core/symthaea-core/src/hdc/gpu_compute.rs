// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WGPU-accelerated linear algebra primitives.
//!
//! Provides high-performance GPU kernels for sparse matrix operations,
//! specifically optimized for HDC and FEM workloads.

/// A GPU-backed sparse matrix solver context.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a new GPU context.
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Symthaea GpuContext"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .ok()?;

        Some(Self { device, queue })
    }
}

/// Sparse Matrix-Vector multiplication (SpMV) kernel in CSR format.
///
/// Shader expects:
/// - row_ptr: array<u32>
/// - col_indices: array<u32>
/// - values: array<f32>
/// - x: array<f32>
/// - result: array<f32> (output)
pub const SPVM_CSR_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= arrayLength(&result)) {
        return;
    }

    let start = row_ptr[row];
    let end = row_ptr[row + 1];
    var sum: f32 = 0.0;

    for (var i = start; i < end; i = i + 1u) {
        let col = col_indices[i];
        let val = values[i];
        sum = sum + val * x[col];
    }

    result[row] = sum;
}
"#;
