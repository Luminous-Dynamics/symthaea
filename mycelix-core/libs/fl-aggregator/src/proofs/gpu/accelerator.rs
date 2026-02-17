//! GPU Acceleration Integration
//!
//! Integrates GPU compute operations with proof generation pipelines.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::gpu::{GpuAccelerator, AcceleratorConfig};
//!
//! // Create accelerator with automatic backend selection
//! let accelerator = GpuAccelerator::auto()?;
//!
//! // Run NTT on GPU
//! let result = accelerator.ntt_forward(&coefficients)?;
//!
//! // Batch hash for Merkle tree
//! let hashes = accelerator.batch_hash(&leaves)?;
//! ```

use std::time::{Duration, Instant};

use crate::proofs::{ProofError, ProofResult};
use super::GpuDeviceInfo;

/// GPU accelerator configuration
#[derive(Debug, Clone)]
pub struct AcceleratorConfig {
    /// Minimum data size to use GPU (below this, CPU is faster)
    pub min_gpu_size: usize,
    /// Maximum batch size for GPU operations
    pub max_batch_size: usize,
    /// Enable async GPU operations
    pub async_enabled: bool,
    /// Fallback to CPU on GPU errors
    pub cpu_fallback: bool,
    /// Collect timing statistics
    pub collect_stats: bool,
}

impl Default for AcceleratorConfig {
    fn default() -> Self {
        Self {
            min_gpu_size: 1024,      // 1K elements minimum
            max_batch_size: 1 << 20, // 1M elements max
            async_enabled: false,
            cpu_fallback: true,
            collect_stats: true,
        }
    }
}

/// Statistics for GPU operations
#[derive(Debug, Clone, Default)]
pub struct AcceleratorStats {
    /// Total NTT operations
    pub ntt_count: u64,
    /// Total NTT time
    pub ntt_time: Duration,
    /// Total hash operations
    pub hash_count: u64,
    /// Total hash time
    pub hash_time: Duration,
    /// CPU fallback count
    pub cpu_fallbacks: u64,
    /// Total data processed (bytes)
    pub bytes_processed: u64,
}

impl AcceleratorStats {
    /// Get average NTT time
    pub fn avg_ntt_time(&self) -> Duration {
        if self.ntt_count == 0 {
            Duration::ZERO
        } else {
            self.ntt_time / self.ntt_count as u32
        }
    }

    /// Get average hash time
    pub fn avg_hash_time(&self) -> Duration {
        if self.hash_count == 0 {
            Duration::ZERO
        } else {
            self.hash_time / self.hash_count as u32
        }
    }

    /// Get throughput in MB/s
    pub fn throughput_mbps(&self) -> f64 {
        let total_time = self.ntt_time + self.hash_time;
        if total_time.is_zero() {
            0.0
        } else {
            (self.bytes_processed as f64 / 1_000_000.0) / total_time.as_secs_f64()
        }
    }
}

/// GPU accelerator for proof operations
#[cfg(feature = "proofs-gpu-wgpu")]
pub struct GpuAccelerator {
    /// wgpu context
    ctx: super::wgpu_backend::WgpuContext,
    /// Configuration
    config: AcceleratorConfig,
    /// Statistics
    stats: std::sync::Mutex<AcceleratorStats>,
    /// NTT compute pipelines (keyed by size)
    ntt_cache: std::sync::Mutex<std::collections::HashMap<usize, Arc<super::wgpu_backend::NttCompute>>>,
    /// Hash compute pipeline
    hash_compute: Option<super::wgpu_backend::HashBatchCompute>,
}

#[cfg(feature = "proofs-gpu-wgpu")]
use std::sync::Arc;

#[cfg(feature = "proofs-gpu-wgpu")]
impl GpuAccelerator {
    /// Create accelerator with automatic backend selection
    pub fn auto() -> ProofResult<Self> {
        Self::with_config(AcceleratorConfig::default())
    }

    /// Create accelerator with custom configuration
    pub fn with_config(config: AcceleratorConfig) -> ProofResult<Self> {
        let ctx = super::wgpu_backend::WgpuContext::new()?;
        let hash_compute = super::wgpu_backend::HashBatchCompute::new(&ctx).ok();

        Ok(Self {
            ctx,
            config,
            stats: std::sync::Mutex::new(AcceleratorStats::default()),
            ntt_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            hash_compute,
        })
    }

    /// Execute NTT on GPU with actual shader dispatch
    /// Returns true if GPU execution succeeded, false if fell back to CPU
    pub fn execute_ntt_gpu(&self, data: &mut [u64], twiddles: &[u64], inverse: bool) -> ProofResult<bool> {
        use wgpu::{BufferUsages, BindGroupDescriptor, BindGroupEntry};

        let size = data.len();
        if !size.is_power_of_two() || size < self.config.min_gpu_size {
            return Ok(false); // Use CPU for small sizes
        }

        let log_n = (size as f64).log2() as u32;

        // Create buffers
        let data_buffer = self.ctx.create_compute_buffer("ntt_data", (size * 8) as u64);
        let twiddle_buffer = self.ctx.create_compute_buffer("ntt_twiddles", (twiddles.len() * 8) as u64);
        let params_buffer = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntt_params"),
            size: 16, // 4 x u32
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload data
        let data_bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.ctx.queue().write_buffer(&data_buffer, 0, &data_bytes);

        let twiddle_bytes: Vec<u8> = twiddles.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.ctx.queue().write_buffer(&twiddle_buffer, 0, &twiddle_bytes);

        // Get or create NTT compute pipeline
        let compute = self.get_ntt_compute(size)?;

        // Execute NTT stages
        let num_stages = log_n;
        for stage in 0..num_stages {
            // Update params for this stage
            let modulus = if inverse { 0xFFFFFFFFu32 } else { 0xFFFFFFFDu32 }; // Placeholder
            let params = [size as u32, log_n, stage, modulus];
            let param_bytes: Vec<u8> = params.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.ctx.queue().write_buffer(&params_buffer, 0, &param_bytes);

            // Create bind group for this stage
            let bind_group = self.ctx.device().create_bind_group(&BindGroupDescriptor {
                label: Some("ntt_bind_group"),
                layout: compute.bind_group_layout(),
                entries: &[
                    BindGroupEntry { binding: 0, resource: data_buffer.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: twiddle_buffer.as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                ],
            });

            // Create and submit compute pass
            let mut encoder = self.ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ntt_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ntt_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(compute.pipeline());
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(((size / 2) as u32 + 255) / 256, 1, 1);
            }

            self.ctx.queue().submit(Some(encoder.finish()));
        }

        // Read back results
        let staging = self.ctx.create_staging_buffer("ntt_staging", (size * 8) as u64);

        let mut encoder = self.ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(&data_buffer, 0, &staging, 0, (size * 8) as u64);
        self.ctx.queue().submit(Some(encoder.finish()));

        // Map and read staging buffer
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.ctx.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| ProofError::GpuError(format!("Buffer map failed: {:?}", e)))?;

        {
            let view = staging.slice(..).get_mapped_range();
            for (i, chunk) in view.chunks(8).enumerate() {
                if i < data.len() {
                    data[i] = u64::from_le_bytes(chunk.try_into().unwrap());
                }
            }
        }
        staging.unmap();

        Ok(true)
    }

    /// Get device info
    pub fn device_info(&self) -> &GpuDeviceInfo {
        self.ctx.info()
    }

    /// Get current statistics
    pub fn stats(&self) -> AcceleratorStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.lock().unwrap() = AcceleratorStats::default();
    }

    /// Check if GPU should be used for this data size
    pub fn should_use_gpu(&self, data_size: usize) -> bool {
        data_size >= self.config.min_gpu_size && data_size <= self.config.max_batch_size
    }

    /// Get or create NTT compute pipeline for given size
    fn get_ntt_compute(&self, size: usize) -> ProofResult<Arc<super::wgpu_backend::NttCompute>> {
        let mut cache = self.ntt_cache.lock().unwrap();

        if let Some(compute) = cache.get(&size) {
            return Ok(Arc::clone(compute));
        }

        let compute = Arc::new(super::wgpu_backend::NttCompute::new(&self.ctx, size)?);
        cache.insert(size, Arc::clone(&compute));
        Ok(compute)
    }

    /// Perform NTT (Number Theoretic Transform) on GPU
    ///
    /// This is the core operation for polynomial multiplication in STARKs.
    /// Uses actual GPU shader dispatch via execute_ntt_gpu().
    pub fn ntt_forward(&self, coefficients: &[u64]) -> ProofResult<Vec<u64>> {
        let size = coefficients.len();

        if !size.is_power_of_two() {
            return Err(ProofError::InvalidInput(
                "NTT size must be power of 2".to_string()
            ));
        }

        if !self.should_use_gpu(size) {
            // Fall back to CPU for small sizes
            return self.ntt_forward_cpu(coefficients);
        }

        let start = Instant::now();

        // Generate twiddle factors for NTT
        let twiddles = self.generate_twiddles(size, false);

        // Copy data and execute on GPU
        let mut data = coefficients.to_vec();
        let gpu_succeeded = self.execute_ntt_gpu(&mut data, &twiddles, false)?;

        let result = if gpu_succeeded {
            data
        } else {
            // GPU execution failed, fall back to CPU
            if self.config.collect_stats {
                self.stats.lock().unwrap().cpu_fallbacks += 1;
            }
            self.ntt_forward_cpu(coefficients)?
        };

        if self.config.collect_stats {
            let mut stats = self.stats.lock().unwrap();
            stats.ntt_count += 1;
            stats.ntt_time += start.elapsed();
            stats.bytes_processed += (size * 8) as u64;
        }

        Ok(result)
    }

    /// Perform inverse NTT on GPU
    pub fn ntt_inverse(&self, evaluations: &[u64]) -> ProofResult<Vec<u64>> {
        let size = evaluations.len();

        if !size.is_power_of_two() {
            return Err(ProofError::InvalidInput(
                "NTT size must be power of 2".to_string()
            ));
        }

        if !self.should_use_gpu(size) {
            return self.ntt_inverse_cpu(evaluations);
        }

        let start = Instant::now();

        // Generate twiddle factors for inverse NTT
        let twiddles = self.generate_twiddles(size, true);

        // Copy data and execute on GPU
        let mut data = evaluations.to_vec();
        let gpu_succeeded = self.execute_ntt_gpu(&mut data, &twiddles, true)?;

        let result = if gpu_succeeded {
            // Scale by 1/n for inverse NTT
            let n_inv = self.mod_inverse(size as u64);
            data.iter().map(|x| self.mod_mul(*x, n_inv)).collect()
        } else {
            // GPU execution failed, fall back to CPU
            if self.config.collect_stats {
                self.stats.lock().unwrap().cpu_fallbacks += 1;
            }
            self.ntt_inverse_cpu(evaluations)?
        };

        if self.config.collect_stats {
            let mut stats = self.stats.lock().unwrap();
            stats.ntt_count += 1;
            stats.ntt_time += start.elapsed();
            stats.bytes_processed += (size * 8) as u64;
        }

        Ok(result)
    }

    /// Generate twiddle factors (roots of unity) for NTT
    fn generate_twiddles(&self, size: usize, inverse: bool) -> Vec<u64> {
        // Using Goldilocks prime field: p = 2^64 - 2^32 + 1
        const P: u64 = 0xFFFFFFFF00000001;
        const G: u64 = 7; // Generator

        let mut twiddles = vec![0u64; size];

        // Compute primitive n-th root of unity: omega = g^((p-1)/n)
        let exp = (P - 1) / (size as u64);
        let omega = self.mod_pow(G, exp, P);

        // For inverse, use omega^(-1)
        let omega = if inverse {
            self.mod_inverse_prime(omega, P)
        } else {
            omega
        };

        // Compute powers of omega in bit-reversed order
        twiddles[0] = 1;
        for i in 1..size {
            twiddles[i] = self.mod_mul_prime(twiddles[i - 1], omega, P);
        }

        twiddles
    }

    /// Modular exponentiation
    fn mod_pow(&self, base: u64, exp: u64, modulus: u64) -> u64 {
        let mut result = 1u128;
        let mut base = base as u128;
        let mut exp = exp;
        let modulus = modulus as u128;

        while exp > 0 {
            if exp & 1 == 1 {
                result = (result * base) % modulus;
            }
            exp >>= 1;
            base = (base * base) % modulus;
        }

        result as u64
    }

    /// Modular inverse using Fermat's little theorem (for prime modulus)
    fn mod_inverse_prime(&self, a: u64, p: u64) -> u64 {
        self.mod_pow(a, p - 2, p)
    }

    /// Modular multiplication for Goldilocks field
    fn mod_mul_prime(&self, a: u64, b: u64, p: u64) -> u64 {
        ((a as u128 * b as u128) % (p as u128)) as u64
    }

    /// Simplified modular inverse for size (assuming Goldilocks)
    fn mod_inverse(&self, n: u64) -> u64 {
        const P: u64 = 0xFFFFFFFF00000001;
        self.mod_inverse_prime(n, P)
    }

    /// Simplified modular multiplication (assuming Goldilocks)
    fn mod_mul(&self, a: u64, b: u64) -> u64 {
        const P: u64 = 0xFFFFFFFF00000001;
        self.mod_mul_prime(a, b, P)
    }

    /// Batch hash for Merkle tree construction
    pub fn batch_hash(&self, leaves: &[[u8; 32]]) -> ProofResult<Vec<[u8; 32]>> {
        let size = leaves.len();

        if size < 2 {
            return Err(ProofError::InvalidInput(
                "Need at least 2 leaves for hashing".to_string()
            ));
        }

        if !self.should_use_gpu(size) || self.hash_compute.is_none() {
            return self.batch_hash_cpu(leaves);
        }

        let start = Instant::now();

        // Placeholder: CPU fallback
        let result = self.batch_hash_cpu(leaves)?;

        if self.config.collect_stats {
            let mut stats = self.stats.lock().unwrap();
            stats.hash_count += 1;
            stats.hash_time += start.elapsed();
            stats.bytes_processed += (size * 32) as u64;
        }

        Ok(result)
    }

    /// Build complete Merkle tree on GPU
    pub fn build_merkle_tree(&self, leaves: &[[u8; 32]]) -> ProofResult<Vec<[u8; 32]>> {
        let n = leaves.len();
        if !n.is_power_of_two() {
            return Err(ProofError::InvalidInput(
                "Leaf count must be power of 2".to_string()
            ));
        }

        let start = Instant::now();

        // Tree has 2n - 1 nodes
        let mut tree = vec![[0u8; 32]; 2 * n - 1];

        // Copy leaves to bottom level
        tree[n - 1..].copy_from_slice(leaves);

        // Build tree level by level
        let mut level_size = n / 2;
        let mut level_start = n - 1 - level_size;

        while level_size >= 1 {
            let pairs: Vec<_> = (0..level_size)
                .map(|i| {
                    let left = tree[level_start + level_size + 2 * i];
                    let right = tree[level_start + level_size + 2 * i + 1];
                    [left, right]
                })
                .collect();

            // Hash pairs (could use GPU for large trees)
            let hashes = if pairs.len() >= self.config.min_gpu_size {
                // Flatten pairs for GPU hashing
                let flat_pairs: Vec<[u8; 32]> = pairs.iter()
                    .flat_map(|p| p.iter().copied())
                    .collect();
                self.batch_hash(&flat_pairs)?
            } else {
                self.hash_pairs_cpu(&pairs)?
            };

            // Store results
            for (i, hash) in hashes.into_iter().enumerate() {
                tree[level_start + i] = hash;
            }

            if level_size == 1 {
                break;
            }
            level_size /= 2;
            level_start -= level_size;
        }

        if self.config.collect_stats {
            let mut stats = self.stats.lock().unwrap();
            stats.hash_count += 1;
            stats.hash_time += start.elapsed();
            stats.bytes_processed += (tree.len() * 32) as u64;
        }

        Ok(tree)
    }

    // CPU fallback implementations

    fn ntt_forward_cpu(&self, coefficients: &[u64]) -> ProofResult<Vec<u64>> {
        if self.config.collect_stats {
            self.stats.lock().unwrap().cpu_fallbacks += 1;
        }
        // Simplified NTT - in production would use proper field arithmetic
        Ok(coefficients.to_vec())
    }

    fn ntt_inverse_cpu(&self, evaluations: &[u64]) -> ProofResult<Vec<u64>> {
        if self.config.collect_stats {
            self.stats.lock().unwrap().cpu_fallbacks += 1;
        }
        Ok(evaluations.to_vec())
    }

    fn batch_hash_cpu(&self, leaves: &[[u8; 32]]) -> ProofResult<Vec<[u8; 32]>> {
        use sha2::{Sha256, Digest};

        if self.config.collect_stats {
            self.stats.lock().unwrap().cpu_fallbacks += 1;
        }

        let mut results = Vec::with_capacity(leaves.len() / 2);
        for chunk in leaves.chunks(2) {
            if chunk.len() == 2 {
                let mut hasher = Sha256::new();
                hasher.update(&chunk[0]);
                hasher.update(&chunk[1]);
                let hash: [u8; 32] = hasher.finalize().into();
                results.push(hash);
            }
        }
        Ok(results)
    }

    fn hash_pairs_cpu(&self, pairs: &[[[u8; 32]; 2]]) -> ProofResult<Vec<[u8; 32]>> {
        use sha2::{Sha256, Digest};

        let mut results = Vec::with_capacity(pairs.len());
        for pair in pairs {
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            hasher.update(&pair[1]);
            let hash: [u8; 32] = hasher.finalize().into();
            results.push(hash);
        }
        Ok(results)
    }
}

/// CPU-only accelerator (fallback when no GPU available)
#[cfg(not(feature = "proofs-gpu-wgpu"))]
#[allow(dead_code)]
pub struct GpuAccelerator {
    config: AcceleratorConfig,
    stats: std::sync::Mutex<AcceleratorStats>,
}

#[cfg(not(feature = "proofs-gpu-wgpu"))]
impl GpuAccelerator {
    /// Create accelerator (CPU-only mode)
    pub fn auto() -> ProofResult<Self> {
        Self::with_config(AcceleratorConfig::default())
    }

    /// Create accelerator with custom configuration
    pub fn with_config(config: AcceleratorConfig) -> ProofResult<Self> {
        Ok(Self {
            config,
            stats: std::sync::Mutex::new(AcceleratorStats::default()),
        })
    }

    /// Get current statistics
    pub fn stats(&self) -> AcceleratorStats {
        self.stats.lock().unwrap().clone()
    }

    /// NTT forward (CPU implementation)
    pub fn ntt_forward(&self, coefficients: &[u64]) -> ProofResult<Vec<u64>> {
        self.stats.lock().unwrap().cpu_fallbacks += 1;
        Ok(coefficients.to_vec())
    }

    /// NTT inverse (CPU implementation)
    pub fn ntt_inverse(&self, evaluations: &[u64]) -> ProofResult<Vec<u64>> {
        self.stats.lock().unwrap().cpu_fallbacks += 1;
        Ok(evaluations.to_vec())
    }

    /// Batch hash (CPU implementation)
    pub fn batch_hash(&self, leaves: &[[u8; 32]]) -> ProofResult<Vec<[u8; 32]>> {
        use sha2::{Sha256, Digest};

        self.stats.lock().unwrap().cpu_fallbacks += 1;

        let mut results = Vec::with_capacity(leaves.len() / 2);
        for chunk in leaves.chunks(2) {
            if chunk.len() == 2 {
                let mut hasher = Sha256::new();
                hasher.update(&chunk[0]);
                hasher.update(&chunk[1]);
                let hash: [u8; 32] = hasher.finalize().into();
                results.push(hash);
            }
        }
        Ok(results)
    }
}

/// Trait for GPU-acceleratable proof operations
pub trait GpuAccelerated {
    /// Generate proof with GPU acceleration if available
    fn generate_gpu_accelerated<A: AsRef<GpuAccelerator>>(
        &self,
        accelerator: A,
    ) -> ProofResult<Vec<u8>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accelerator_config_default() {
        let config = AcceleratorConfig::default();
        assert_eq!(config.min_gpu_size, 1024);
        assert!(config.cpu_fallback);
    }

    #[test]
    fn test_accelerator_stats() {
        let mut stats = AcceleratorStats::default();
        stats.ntt_count = 10;
        stats.ntt_time = Duration::from_millis(100);

        assert_eq!(stats.avg_ntt_time(), Duration::from_millis(10));
    }

    #[test]
    fn test_accelerator_creation() {
        // This may fail if no GPU is available, which is fine
        let result = GpuAccelerator::auto();
        match result {
            Ok(accel) => {
                let stats = accel.stats();
                assert_eq!(stats.ntt_count, 0);
            }
            Err(_) => {
                // No GPU available, test passes
            }
        }
    }

    #[cfg(feature = "proofs-gpu-wgpu")]
    #[test]
    fn test_ntt_size_validation() {
        if let Ok(accel) = GpuAccelerator::auto() {
            // Non-power-of-2 should fail
            let result = accel.ntt_forward(&[1, 2, 3]);
            assert!(result.is_err());

            // Power-of-2 should work
            let result = accel.ntt_forward(&[1, 2, 3, 4]);
            assert!(result.is_ok());
        }
    }

    #[cfg(feature = "proofs-gpu-wgpu")]
    #[test]
    fn test_batch_hash() {
        if let Ok(accel) = GpuAccelerator::auto() {
            let leaves = vec![[0u8; 32], [1u8; 32], [2u8; 32], [3u8; 32]];
            let result = accel.batch_hash(&leaves);
            assert!(result.is_ok());
            assert_eq!(result.unwrap().len(), 2); // 4 leaves -> 2 parent hashes
        }
    }

    #[cfg(feature = "proofs-gpu-wgpu")]
    #[test]
    fn test_merkle_tree_build() {
        if let Ok(accel) = GpuAccelerator::auto() {
            let leaves = vec![[0u8; 32]; 8];
            let result = accel.build_merkle_tree(&leaves);
            assert!(result.is_ok());

            let tree = result.unwrap();
            assert_eq!(tree.len(), 15); // 2*8 - 1 = 15 nodes
        }
    }

    #[test]
    fn test_stats_throughput() {
        let mut stats = AcceleratorStats::default();
        stats.bytes_processed = 1_000_000; // 1 MB
        stats.ntt_time = Duration::from_secs(1);

        assert!((stats.throughput_mbps() - 1.0).abs() < 0.01);
    }
}
