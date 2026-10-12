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
        let instance = wgpu::Instance::default();
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
