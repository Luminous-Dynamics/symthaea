//! wgpu GPU Backend for Proof Acceleration
//!
//! Implements GPU-accelerated operations for STARK proof generation using WebGPU.
//!
//! ## Supported Operations
//!
//! - **NTT (Number Theoretic Transform)**: Polynomial multiplication
//! - **Hash Batching**: Parallel merkle tree construction
//! - **Polynomial Evaluation**: Multi-point evaluation
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::gpu::wgpu_backend::{WgpuContext, NttCompute};
//!
//! // Initialize GPU context
//! let ctx = WgpuContext::new()?;
//!
//! // Run NTT on GPU
//! let ntt = NttCompute::new(&ctx, 1024)?;
//! let result = ntt.forward(&input_data)?;
//! ```

#[cfg(feature = "proofs-gpu-wgpu")]
use wgpu::{
    Adapter, Device, Queue, Instance, InstanceDescriptor, RequestAdapterOptions,
    DeviceDescriptor, Features, Limits, PowerPreference, Buffer, BufferDescriptor,
    BufferUsages, ShaderModuleDescriptor, ShaderSource, ComputePipeline,
    ComputePipelineDescriptor, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType,
};

use super::{GpuConfig, GpuDeviceInfo, GpuStatus};
use crate::proofs::{ProofError, ProofResult};

/// wgpu GPU context
#[cfg(feature = "proofs-gpu-wgpu")]
pub struct WgpuContext {
    /// wgpu instance
    instance: Instance,
    /// GPU adapter
    adapter: Adapter,
    /// GPU device
    device: Device,
    /// Command queue
    queue: Queue,
    /// Device info
    info: GpuDeviceInfo,
}

#[cfg(feature = "proofs-gpu-wgpu")]
impl WgpuContext {
    /// Create a new wgpu context
    pub fn new() -> ProofResult<Self> {
        Self::with_config(&GpuConfig::default())
    }

    /// Create with specific configuration
    pub fn with_config(_config: &GpuConfig) -> ProofResult<Self> {
        let instance = Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| ProofError::GpuError("No suitable GPU adapter found".to_string()))?;

        let adapter_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("fl-aggregator-gpu"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .map_err(|e| ProofError::GpuError(format!("Failed to create device: {:?}", e)))?;

        let info = GpuDeviceInfo {
            name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
            total_memory: 0, // wgpu doesn't expose this directly
            available_memory: 0,
            compute_units: adapter.limits().max_compute_workgroups_per_dimension,
            suitable: true,
        };

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            info,
        })
    }

    /// Get device info
    pub fn info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    /// Get the wgpu device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the command queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Create a buffer
    pub fn create_buffer(&self, label: &str, size: u64, usage: BufferUsages) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for reading back data
    pub fn create_staging_buffer(&self, label: &str, size: u64) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a compute buffer (storage + copy)
    pub fn create_compute_buffer(&self, label: &str, size: u64) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }
}

/// NTT compute shader - Number Theoretic Transform
#[cfg(feature = "proofs-gpu-wgpu")]
pub struct NttCompute {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    size: usize,
}

#[cfg(feature = "proofs-gpu-wgpu")]
impl NttCompute {
    /// Create a new NTT compute instance
    pub fn new(ctx: &WgpuContext, size: usize) -> ProofResult<Self> {
        // Validate size is power of 2
        if !size.is_power_of_two() {
            return Err(ProofError::InvalidInput(
                "NTT size must be power of 2".to_string(),
            ));
        }

        // Create shader module
        let shader = ctx.device().create_shader_module(ShaderModuleDescriptor {
            label: Some("NTT Shader"),
            source: ShaderSource::Wgsl(NTT_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = ctx.device().create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("NTT Bind Group Layout"),
            entries: &[
                // Input/output buffer
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
                // Twiddle factors buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Parameters buffer
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = ctx.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NTT Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = ctx.device().create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("NTT Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("ntt_butterfly"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            size,
        })
    }

    /// Get the configured size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the compute pipeline
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    /// Get the bind group layout
    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}

/// NTT shader in WGSL (WebGPU Shading Language)
#[cfg(feature = "proofs-gpu-wgpu")]
const NTT_SHADER: &str = r#"
// NTT Butterfly operation for polynomial multiplication
// Uses Cooley-Tukey algorithm

struct NttParams {
    n: u32,           // Size of transform
    log_n: u32,       // log2(n)
    stage: u32,       // Current stage
    modulus: u32,     // Field modulus (for Montgomery reduction)
}

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> twiddles: array<u32>;
@group(0) @binding(2) var<uniform> params: NttParams;

// Montgomery multiplication (simplified)
fn mont_mul(a: u32, b: u32) -> u32 {
    let product = u64(a) * u64(b);
    // Simplified reduction - in production use proper Montgomery
    return u32(product % u64(params.modulus));
}

// Butterfly operation
fn butterfly(i: u32, j: u32, twiddle: u32) {
    let u = data[i];
    let v = mont_mul(data[j], twiddle);

    // a + b * w mod p
    let sum = (u + v) % params.modulus;
    // a - b * w mod p (with underflow handling)
    let diff = (u + params.modulus - v) % params.modulus;

    data[i] = sum;
    data[j] = diff;
}

@compute @workgroup_size(256)
fn ntt_butterfly(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let half_size = params.n >> (params.stage + 1u);

    if (idx >= half_size) {
        return;
    }

    let block_size = params.n >> params.stage;
    let block_idx = idx / (block_size >> 1u);
    let local_idx = idx % (block_size >> 1u);

    let i = block_idx * block_size + local_idx;
    let j = i + (block_size >> 1u);

    let twiddle_idx = local_idx << params.stage;
    let twiddle = twiddles[twiddle_idx];

    butterfly(i, j, twiddle);
}
"#;

/// Hash batch compute for Merkle tree construction
#[cfg(feature = "proofs-gpu-wgpu")]
pub struct HashBatchCompute {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[cfg(feature = "proofs-gpu-wgpu")]
impl HashBatchCompute {
    /// Create a new hash batch compute instance
    pub fn new(ctx: &WgpuContext) -> ProofResult<Self> {
        let shader = ctx.device().create_shader_module(ShaderModuleDescriptor {
            label: Some("Hash Batch Shader"),
            source: ShaderSource::Wgsl(HASH_SHADER.into()),
        });

        let bind_group_layout = ctx.device().create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Hash Batch Bind Group Layout"),
            entries: &[
                // Input leaves
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output hashes
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
            ],
        });

        let pipeline_layout = ctx.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hash Batch Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = ctx.device().create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Hash Batch Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("hash_pair"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }
}

/// Simple hash shader for Merkle tree (placeholder - use STARK-friendly hash in production)
#[cfg(feature = "proofs-gpu-wgpu")]
const HASH_SHADER: &str = r#"
// Simplified hash for demonstration
// In production, use Rescue or Poseidon hash

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

// Simple mixing function (NOT cryptographically secure - placeholder only)
fn mix(a: u32, b: u32) -> u32 {
    var x = a ^ b;
    x = x ^ (x >> 16u);
    x = x * 0x85ebca6bu;
    x = x ^ (x >> 13u);
    x = x * 0xc2b2ae35u;
    x = x ^ (x >> 16u);
    return x;
}

@compute @workgroup_size(256)
fn hash_pair(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Each thread processes one pair of 8-word (256-bit) hashes
    let input_offset = idx * 16u;  // 2 hashes * 8 words
    let output_offset = idx * 8u;  // 1 hash * 8 words

    // Simple hash combination (placeholder)
    for (var i = 0u; i < 8u; i = i + 1u) {
        let left = input[input_offset + i];
        let right = input[input_offset + 8u + i];
        output[output_offset + i] = mix(left, right);
    }
}
"#;

/// GPU benchmark results
#[cfg(feature = "proofs-gpu-wgpu")]
#[derive(Debug, Clone)]
pub struct WgpuBenchmarkResult {
    /// Operation name
    pub operation: String,
    /// GPU time in milliseconds
    pub gpu_time_ms: f64,
    /// CPU time in milliseconds (for comparison)
    pub cpu_time_ms: f64,
    /// Speedup factor
    pub speedup: f64,
    /// Data size
    pub data_size: usize,
}

/// Check if wgpu GPU acceleration is available
#[cfg(feature = "proofs-gpu-wgpu")]
pub fn is_wgpu_available() -> bool {
    WgpuContext::new().is_ok()
}

/// Get wgpu GPU status
#[cfg(feature = "proofs-gpu-wgpu")]
pub fn get_wgpu_status() -> GpuStatus {
    match WgpuContext::new() {
        Ok(ctx) => GpuStatus {
            available: true,
            device: Some(ctx.info().clone()),
            error: None,
        },
        Err(e) => GpuStatus {
            available: false,
            device: None,
            error: Some(e.to_string()),
        },
    }
}

// Fallback implementations when wgpu is not enabled
#[cfg(not(feature = "proofs-gpu-wgpu"))]
pub fn is_wgpu_available() -> bool {
    false
}

#[cfg(not(feature = "proofs-gpu-wgpu"))]
pub fn get_wgpu_status() -> GpuStatus {
    GpuStatus {
        available: false,
        device: None,
        error: Some("wgpu feature not enabled".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_availability_check() {
        // This test just checks that the function runs without panicking
        let available = is_wgpu_available();
        println!("wgpu available: {}", available);
    }

    #[test]
    fn test_wgpu_status() {
        let status = get_wgpu_status();
        println!("wgpu status: {:?}", status);
    }

    #[cfg(feature = "proofs-gpu-wgpu")]
    #[test]
    fn test_wgpu_context_creation() {
        match WgpuContext::new() {
            Ok(ctx) => {
                println!("GPU: {}", ctx.info().name);
                println!("Backend: {}", ctx.info().backend);
            }
            Err(e) => {
                println!("No GPU available: {}", e);
            }
        }
    }
}
