// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! WGSL compute backend for field diffusion.
//!
//! The CPU stepper remains the reference implementation. This backend exists to
//! prove the shader path against the same `FieldStepper` contract.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{FieldGrid, FieldLayer, FieldStepError, FieldStepRequest, FieldStepper};

const SHADER: &str = r#"
struct Params {
    width: u32,
    height: u32,
    diffusion: f32,
    decay: f32,
    dt: f32,
    max_value: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> input_field: array<f32>;
@group(0) @binding(1) var<storage, read> obstacle_field: array<f32>;
@group(0) @binding(2) var<storage, read> source_field: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_field: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn idx(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

fn sanitize_concentration(value: f32) -> f32 {
    if !(value == value) {
        return 0.0;
    }
    return clamp(value, 0.0, params.max_value);
}

fn sanitize_source(value: f32) -> f32 {
    if !(value == value) {
        return 0.0;
    }
    return max(value, 0.0);
}

fn neighbor(x: i32, y: i32, fallback: f32) -> f32 {
    if x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.height) {
        return fallback;
    }
    let offset = idx(u32(x), u32(y));
    if obstacle_field[offset] >= 0.5 {
        return fallback;
    }
    return input_field[offset];
}

@compute @workgroup_size(8, 8, 1)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }

    let offset = idx(id.x, id.y);
    if obstacle_field[offset] >= 0.5 {
        output_field[offset] = 0.0;
        return;
    }

    let center = input_field[offset];
    let x = i32(id.x);
    let y = i32(id.y);
    let left = neighbor(x - 1, y, center);
    let right = neighbor(x + 1, y, center);
    let up = neighbor(x, y - 1, center);
    let down = neighbor(x, y + 1, center);
    let laplacian = left + right + up + down - 4.0 * center;
    let source = sanitize_source(source_field[offset]);
    let next = center + params.dt * (params.diffusion * laplacian - params.decay * center + source);
    output_field[offset] = sanitize_concentration(next);
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    diffusion: f32,
    decay: f32,
    dt: f32,
    max_value: f32,
    _pad0: u32,
    _pad1: u32,
}

/// Single-dispatch WGSL implementation of `FieldStepper`.
///
/// This is intentionally simple: create buffers, dispatch one compute pass,
/// read back the selected layer. Long-lived buffer reuse can be added once the
/// parity contract is stable.
#[derive(Debug, Clone, Copy, Default)]
pub struct WgslFieldStepper;

impl FieldStepper for WgslFieldStepper {
    fn step(
        &self,
        field: &mut FieldGrid,
        request: &FieldStepRequest,
    ) -> Result<(), FieldStepError> {
        pollster::block_on(step_async(field, request))
    }
}

async fn step_async(
    field: &mut FieldGrid,
    request: &FieldStepRequest,
) -> Result<(), FieldStepError> {
    request.params.validate()?;
    let cell_count = field.width() * field.height();
    if request.source.len() != cell_count {
        return Err(FieldStepError::GpuDispatchFailed(format!(
            "source length {} does not match field cell count {cell_count}",
            request.source.len()
        )));
    }

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| FieldStepError::GpuUnavailable(e.to_string()))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("symtropy-lifesim-core"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| FieldStepError::GpuUnavailable(e.to_string()))?;

    let input = &field.channels[request.layer.index()];
    let obstacle = &field.channels[FieldLayer::Obstacle.index()];
    let output = vec![0.0f32; cell_count];
    let params = GpuParams {
        width: field.width() as u32,
        height: field.height() as u32,
        diffusion: request.params.diffusion,
        decay: request.params.decay,
        dt: request.params.dt,
        max_value: request.params.max_value,
        _pad0: 0,
        _pad1: 0,
    };

    let input_buffer = storage_buffer(&device, "field-input", input, wgpu::BufferUsages::STORAGE);
    let obstacle_buffer = storage_buffer(
        &device,
        "field-obstacle",
        obstacle,
        wgpu::BufferUsages::STORAGE,
    );
    let source_buffer = storage_buffer(
        &device,
        "field-source",
        &request.source,
        wgpu::BufferUsages::STORAGE,
    );
    let output_buffer = storage_buffer(
        &device,
        "field-output",
        &output,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("field-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("field-readback"),
        size: std::mem::size_of_val(output.as_slice()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("field-diffuse-decay"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("field-bind-group-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("field-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("field-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("step"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("field-bind-group"),
        layout: &bind_group_layout,
        entries: &[
            bind_entry(0, &input_buffer),
            bind_entry(1, &obstacle_buffer),
            bind_entry(2, &source_buffer),
            bind_entry(3, &output_buffer),
            bind_entry(4, &params_buffer),
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("field-command-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("field-compute-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            (field.width() as u32).div_ceil(8),
            (field.height() as u32).div_ceil(8),
            1,
        );
    }
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &readback_buffer,
        0,
        std::mem::size_of_val(output.as_slice()) as u64,
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback_buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| FieldStepError::GpuDispatchFailed(e.to_string()))?;
    rx.recv()
        .map_err(|e| FieldStepError::GpuDispatchFailed(e.to_string()))?
        .map_err(|e| FieldStepError::GpuDispatchFailed(e.to_string()))?;

    let mapped = slice.get_mapped_range();
    let values: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback_buffer.unmap();

    field.channels[request.layer.index()] = values;
    Ok(())
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    values: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage,
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CpuFieldStepper, DiffusionParams, FieldLayer, FieldStepRequest,
        compare_layer_within_epsilon,
    };

    #[test]
    fn wgsl_stepper_matches_cpu_reference_with_source_and_obstacles() {
        let mut cpu = FieldGrid::new(8, 6);
        let mut gpu = FieldGrid::new(8, 6);
        for field in [&mut cpu, &mut gpu] {
            field.set(FieldLayer::Nutrient, 3, 3, 15.0);
            field.set(FieldLayer::Nutrient, 5, 1, 2.0);
            field.set(FieldLayer::Obstacle, 4, 3, 1.0);
        }
        let mut source = vec![0.0; cpu.width() * cpu.height()];
        source[cpu.idx(2, 2)] = 4.0;
        let request = FieldStepRequest {
            layer: FieldLayer::Nutrient,
            source,
            params: DiffusionParams {
                diffusion: 0.08,
                decay: 0.02,
                dt: 1.0,
                max_value: 1_000.0,
            },
        };

        CpuFieldStepper.step(&mut cpu, &request).unwrap();
        match WgslFieldStepper.step(&mut gpu, &request) {
            Ok(()) => {}
            Err(FieldStepError::GpuUnavailable(message)) => {
                eprintln!("skipping WGSL parity test: {message}");
                return;
            }
            Err(error) => panic!("WGSL field step failed: {error}"),
        }

        let report = compare_layer_within_epsilon(&cpu, &gpu, FieldLayer::Nutrient, 0.0001);
        assert!(report.within_epsilon(), "report={report:?}");
    }
}
