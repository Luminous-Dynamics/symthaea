// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WGPU-accelerated linear algebra primitives.
//!
//! Provides high-performance GPU kernels for sparse matrix operations,
//! specifically optimized for HDC and FEM workloads.

use serde::{Deserialize, Serialize};

/// Runtime adapter identity and the limits available to a created device.
///
/// Values are observations made during initialization, not promises about
/// remaining free memory. Buffer allocation can still fail below these limits.
/// Keep this report with benchmark/effect receipts so the selected backend and
/// its resource envelope are explicit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuAdapterReport {
    pub adapter_name: String,
    pub backend: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
    /// True only when wgpu classifies the adapter as a CPU device.
    /// False does not, by itself, prove physical hardware execution.
    pub is_cpu_device: bool,
    /// True when wgpu reports its non-computing NOOP backend.
    pub is_noop_backend: bool,

    pub adapter_max_buffer_size: u64,
    pub device_max_buffer_size: u64,
    pub adapter_max_storage_buffer_binding_size: u64,
    pub device_max_storage_buffer_binding_size: u64,
    pub adapter_max_compute_workgroups_per_dimension: u32,
    pub device_max_compute_workgroups_per_dimension: u32,
    pub adapter_max_compute_workgroup_size_x: u32,
    pub device_max_compute_workgroup_size_x: u32,
    pub adapter_max_compute_invocations_per_workgroup: u32,
    pub device_max_compute_invocations_per_workgroup: u32,
}

impl GpuAdapterReport {
    fn from_wgpu(
        info: &wgpu::AdapterInfo,
        adapter_limits: &wgpu::Limits,
        device_limits: &wgpu::Limits,
    ) -> Self {
        Self {
            adapter_name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            vendor_id: info.vendor,
            device_id: info.device,
            device_type: format!("{:?}", info.device_type),
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            is_cpu_device: info.device_type == wgpu::DeviceType::Cpu,
            is_noop_backend: info.backend == wgpu::Backend::Noop,
            adapter_max_buffer_size: adapter_limits.max_buffer_size,
            device_max_buffer_size: device_limits.max_buffer_size,
            adapter_max_storage_buffer_binding_size: u64::from(
                adapter_limits.max_storage_buffer_binding_size,
            ),
            device_max_storage_buffer_binding_size: u64::from(
                device_limits.max_storage_buffer_binding_size,
            ),
            adapter_max_compute_workgroups_per_dimension:
                adapter_limits.max_compute_workgroups_per_dimension,
            device_max_compute_workgroups_per_dimension:
                device_limits.max_compute_workgroups_per_dimension,
            adapter_max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
            device_max_compute_workgroup_size_x: device_limits.max_compute_workgroup_size_x,
            adapter_max_compute_invocations_per_workgroup:
                adapter_limits.max_compute_invocations_per_workgroup,
            device_max_compute_invocations_per_workgroup:
                device_limits.max_compute_invocations_per_workgroup,
        }
    }

    /// A conservative diagnostic for obvious software/no-op adapters.
    ///
    /// A false result is not a hardware attestation: virtual adapters and
    /// driver-specific software paths may require an additional policy check.
    pub fn is_obviously_software_or_noop(&self) -> bool {
        self.is_cpu_device || self.is_noop_backend
    }

    /// Check a buffer size against both physical-adapter and created-device limits.
    ///
    /// This is a limit check, not a guarantee that the allocation will succeed;
    /// available memory and fragmentation may change at runtime.
    pub fn supports_buffer_size(&self, bytes: u64) -> bool {
        bytes <= self.adapter_max_buffer_size && bytes <= self.device_max_buffer_size
    }

    /// Check a storage-buffer binding against all relevant reported size limits.
    pub fn supports_storage_binding(&self, bytes: u64) -> bool {
        self.supports_buffer_size(bytes)
            && bytes <= self.adapter_max_storage_buffer_binding_size
            && bytes <= self.device_max_storage_buffer_binding_size
    }

    /// Check the X dimension of a one-dimensional compute dispatch.
    ///
    /// The caller must still calculate workgroup counts with checked arithmetic
    /// and chunk or reject dispatches that exceed this limit.
    pub fn supports_1d_dispatch(&self, workgroups: u32, workgroup_size: u32) -> bool {
        workgroups > 0
            && workgroup_size > 0
            && workgroups
                <= self
                    .adapter_max_compute_workgroups_per_dimension
                    .min(self.device_max_compute_workgroups_per_dimension)
            && workgroup_size
                <= self
                    .adapter_max_compute_workgroup_size_x
                    .min(self.device_max_compute_workgroup_size_x)
            && workgroup_size
                <= self
                    .adapter_max_compute_invocations_per_workgroup
                    .min(self.device_max_compute_invocations_per_workgroup)
    }
}

/// Why a WGPU context could not be initialized.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GpuContextInitError {
    #[error("no compatible GPU adapter: {0}")]
    AdapterUnavailable(String),
    #[error("GPU device creation failed: {0}")]
    DeviceRequestFailed(String),
}

/// A GPU-backed sparse matrix solver context.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a GPU context, preserving the original Option-based API.
    ///
    /// Use `new_with_report` where adapter identity, limits, or error details
    /// must be captured as evidence.
    pub async fn new() -> Option<Self> {
        Self::new_with_report()
            .await
            .ok()
            .map(|(context, _report)| context)
    }

    /// Create a GPU context and return the selected adapter's runtime report.
    ///
    /// This reports which adapter WGPU selected; it does not by itself prove a
    /// compute operation executed successfully or that the adapter is physical
    /// hardware. Callers needing hardware-only execution must check the report
    /// and then verify a real operation/readback.
    pub async fn new_with_report() -> Result<(Self, GpuAdapterReport), GpuContextInitError> {
        // Honor WGPU_BACKEND and related documented WGPU_* runtime settings.
        // This is essential for qualification jobs that must force Vulkan rather
        // than accidentally exercising another enabled backend.
        let instance_descriptor = wgpu::InstanceDescriptor::default().with_env();
        let instance = wgpu::Instance::new(&instance_descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|error| GpuContextInitError::AdapterUnavailable(error.to_string()))?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();

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
            .map_err(|error| GpuContextInitError::DeviceRequestFailed(error.to_string()))?;

        let device_limits = device.limits();
        let report = GpuAdapterReport::from_wgpu(&adapter_info, &adapter_limits, &device_limits);
        Ok((Self { device, queue }, report))
    }
}

#[cfg(test)]
mod capability_report_tests {
    use super::GpuAdapterReport;

    fn fixture() -> GpuAdapterReport {
        GpuAdapterReport {
            adapter_name: "fixture adapter".to_string(),
            backend: "Vulkan".to_string(),
            vendor_id: 1,
            device_id: 2,
            device_type: "DiscreteGpu".to_string(),
            driver: "fixture driver".to_string(),
            driver_info: "fixture version".to_string(),
            is_cpu_device: false,
            is_noop_backend: false,
            adapter_max_buffer_size: 1024,
            device_max_buffer_size: 512,
            adapter_max_storage_buffer_binding_size: 512,
            device_max_storage_buffer_binding_size: 256,
            adapter_max_compute_workgroups_per_dimension: 64,
            device_max_compute_workgroups_per_dimension: 32,
            adapter_max_compute_workgroup_size_x: 256,
            device_max_compute_workgroup_size_x: 128,
            adapter_max_compute_invocations_per_workgroup: 256,
            device_max_compute_invocations_per_workgroup: 128,
        }
    }

    #[test]
    fn buffer_checks_use_the_tighter_adapter_and_device_limit() {
        let report = fixture();
        assert!(report.supports_buffer_size(512));
        assert!(!report.supports_buffer_size(513));
    }

    #[test]
    fn storage_binding_checks_every_relevant_limit() {
        let report = fixture();
        assert!(report.supports_storage_binding(256));
        assert!(!report.supports_storage_binding(257));
        assert!(!report.supports_storage_binding(513));
    }

    #[test]
    fn dispatch_checks_workgroup_and_dispatch_limits() {
        let report = fixture();
        assert!(report.supports_1d_dispatch(32, 128));
        assert!(!report.supports_1d_dispatch(33, 128));
        assert!(!report.supports_1d_dispatch(1, 129));
        assert!(!report.supports_1d_dispatch(0, 1));
        assert!(!report.supports_1d_dispatch(1, 0));
    }

    #[test]
    fn software_detection_does_not_label_every_unknown_adapter_as_hardware() {
        let mut report = fixture();
        assert!(!report.is_obviously_software_or_noop());
        report.is_cpu_device = true;
        assert!(report.is_obviously_software_or_noop());
        report.is_cpu_device = false;
        report.is_noop_backend = true;
        assert!(report.is_obviously_software_or_noop());
    }

    #[test]
    fn report_round_trips_as_json_for_receipts() {
        let report = fixture();
        let json = serde_json::to_string(&report).expect("serialize capability report");
        let decoded: GpuAdapterReport =
            serde_json::from_str(&json).expect("deserialize capability report");
        assert_eq!(decoded, report);
    }

    /// Run in CI only when a software-Vulkan or NVIDIA-hardware lane opts in.
    ///
    /// This dispatches a WGSL compute shader through WGPU/Vulkan, copies the
    /// result to a mappable buffer, and checks the readback. The software lane
    /// explicitly selects Mesa lavapipe; the hardware lane requires NVIDIA.
    /// Passing on software Vulkan never qualifies physical-GPU execution.
    #[tokio::test]
    async fn vulkan_compute_dispatch_and_readback_when_requested() {
        let expect_software = std::env::var_os("SYTHAEA_GPU_EXPECT_SOFTWARE").is_some();
        let expect_nvidia = std::env::var_os("SYTHAEA_GPU_EXPECT_NVIDIA").is_some();
        if !expect_software && !expect_nvidia {
            return;
        }
        assert_ne!(
            expect_software, expect_nvidia,
            "select exactly one GPU qualification mode"
        );

        use std::time::Duration;

        let qualification_mode = if expect_software {
            "Mesa lavapipe software"
        } else {
            "NVIDIA physical GPU"
        };
        println!("Symthaea GPU qualification mode: {qualification_mode}");

        let (context, report) = super::GpuContext::new_with_report()
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "required {qualification_mode} Vulkan adapter must initialize: {error}"
                )
            });

        println!(
            "Symthaea GPU smoke adapter report: {}",
            serde_json::to_string(&report).expect("serialize runtime adapter report")
        );

        assert!(
            report.backend.contains("Vulkan"),
            "expected Vulkan backend, got {}",
            report.backend
        );
        assert!(!report.is_noop_backend, "NOOP backend cannot qualify compute execution");
        if expect_software {
            assert!(report.is_cpu_device, "software-Vulkan lane expected a CPU-classified adapter");
        } else {
            assert!(!report.is_cpu_device, "NVIDIA hardware lane rejected a CPU-classified adapter");
            assert_eq!(report.vendor_id, 0x10DE, "hardware lane expected NVIDIA vendor ID");
        }
        // Exercise Symthaea's actual BinaryHV bind implementation as the CPU
        // oracle. The first 512 words cover one complete 16,384-bit BinaryHV;
        // one extra sentinel word forces a masked partial final workgroup.
        use crate::hdc::binary_hv::BinaryHV;
        const VECTOR_WORDS: usize = BinaryHV::BYTES / std::mem::size_of::<u32>();
        const ELEMENTS: usize = VECTOR_WORDS + 1;
        const BYTES: u64 = (ELEMENTS * std::mem::size_of::<u32>()) as u64;
        let workgroups = (ELEMENTS as u32).div_ceil(64);
        assert!(
            report.supports_storage_binding(BYTES),
            "device report rejects the required {BYTES}-byte storage binding"
        );
        assert!(
            report.supports_1d_dispatch(workgroups, 64),
            "device report rejects {workgroups} workgroups of 64 invocations"
        );

        let left_hv = BinaryHV::random(0x5348_4456);
        let right_hv = BinaryHV::random(0x4844_4342);
        let bound_hv = left_hv.bind(&right_hv);
        let mut left: Vec<u32> = left_hv
            .0
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let mut right: Vec<u32> = right_hv
            .0
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let mut expected: Vec<u32> = bound_hv
            .0
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        left.push(0xA55A_1234);
        right.push(0x5AA5_FEDC);
        expected.push(left[VECTOR_WORDS] ^ right[VECTOR_WORDS]);

        const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> left: array<u32>;
@group(0) @binding(1) var<storage, read> right: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&result)) {
        return;
    }
    result[index] = left[index] ^ right[index];
}
"#;

        let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Symthaea HDC XOR smoke shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Symthaea HDC XOR smoke pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let left_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Symthaea HDC XOR left input"),
            size: BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let right_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Symthaea HDC XOR right input"),
            size: BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Symthaea HDC XOR output"),
            size: BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Symthaea HDC XOR readback"),
            size: BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        context.queue.write_buffer(&left_buffer, 0, bytemuck::cast_slice(&left));
        context.queue.write_buffer(&right_buffer, 0, bytemuck::cast_slice(&right));

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Symthaea HDC XOR bindings"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: left_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: right_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Symthaea HDC XOR commands"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Symthaea HDC XOR pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, BYTES);
        context.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        context
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .expect("Vulkan compute/readback submission should complete");
        receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("Vulkan readback callback should complete")
            .expect("Vulkan readback buffer should map");

        let bytes = readback.slice(..).get_mapped_range();
        let actual: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(actual.len(), ELEMENTS);
        assert_eq!(actual, expected, "GPU HDC XOR output differs from CPU oracle");
        drop(bytes);
        readback.unmap();
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
    let end = row_ptr[row + 1u];
    var sum: f32 = 0.0;

    for (var i = start; i < end; i = i + 1u) {
        let col = col_indices[i];
        let val = values[i];
        sum = sum + val * x[col];
    }

    result[row] = sum;
}
"#;
