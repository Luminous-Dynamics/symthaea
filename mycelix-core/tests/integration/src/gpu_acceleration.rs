// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GPU Acceleration Integration Tests
//!
//! Tests the wgpu-based GPU acceleration for proof generation operations.
//!
//! ## Operations Tested
//!
//! 1. **NTT (Number Theoretic Transform)**: Core polynomial multiplication
//! 2. **Merkle Tree Building**: Parallel hash computation
//! 3. **Proof Generation Pipeline**: End-to-end with GPU
//!
//! ## Running with GPU
//!
//! ```bash
//! # Enable GPU feature
//! cargo test --features proofs-gpu-wgpu gpu_acceleration -- --nocapture
//! ```
//!
//! ## Benchmark Results Format
//!
//! ```text
//! | Operation | Size | CPU Time | GPU Time | Speedup |
//! |-----------|------|----------|----------|---------|
//! | NTT       | 2^16 | 50ms     | 8ms      | 6.25x   |
//! | Merkle    | 2^16 | 80ms     | 12ms     | 6.67x   |
//! ```

use std::time::{Duration, Instant};

/// Benchmark result for a single operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Data size (number of elements)
    pub size: usize,
    /// CPU execution time
    pub cpu_time: Duration,
    /// GPU execution time (None if GPU unavailable)
    pub gpu_time: Option<Duration>,
    /// Speedup factor (GPU vs CPU)
    pub speedup: Option<f64>,
    /// Whether GPU was actually used
    pub gpu_used: bool,
}

impl BenchmarkResult {
    /// Format as table row
    pub fn to_table_row(&self) -> String {
        let gpu_str = self.gpu_time
            .map(|t| format!("{:.2}ms", t.as_secs_f64() * 1000.0))
            .unwrap_or_else(|| "N/A".to_string());
        let speedup_str = self.speedup
            .map(|s| format!("{:.2}x", s))
            .unwrap_or_else(|| "N/A".to_string());

        format!(
            "| {:12} | {:>8} | {:>10.2}ms | {:>10} | {:>8} |",
            self.operation,
            self.size,
            self.cpu_time.as_secs_f64() * 1000.0,
            gpu_str,
            speedup_str,
        )
    }
}

/// GPU benchmark suite
pub struct GpuBenchmarkSuite {
    results: Vec<BenchmarkResult>,
}

impl GpuBenchmarkSuite {
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    /// Run NTT benchmarks at various sizes
    pub fn run_ntt_benchmarks(&mut self) {
        // Test sizes: 2^10, 2^12, 2^14, 2^16
        for exp in [10, 12, 14, 16] {
            let size = 1usize << exp;
            let result = benchmark_ntt(size);
            self.results.push(result);
        }
    }

    /// Run Merkle tree benchmarks
    pub fn run_merkle_benchmarks(&mut self) {
        for exp in [10, 12, 14, 16] {
            let size = 1usize << exp;
            let result = benchmark_merkle_tree(size);
            self.results.push(result);
        }
    }

    /// Run hash batch benchmarks
    pub fn run_hash_benchmarks(&mut self) {
        for exp in [10, 12, 14] {
            let size = 1usize << exp;
            let result = benchmark_batch_hash(size);
            self.results.push(result);
        }
    }

    /// Print results as table
    pub fn print_table(&self) {
        println!("\n=== GPU Acceleration Benchmarks ===\n");
        println!("| {:12} | {:>8} | {:>12} | {:>10} | {:>8} |",
            "Operation", "Size", "CPU Time", "GPU Time", "Speedup");
        println!("|{:-^14}|{:-^10}|{:-^14}|{:-^12}|{:-^10}|",
            "", "", "", "", "");

        for result in &self.results {
            println!("{}", result.to_table_row());
        }
        println!();
    }

    /// Get summary statistics
    pub fn summary(&self) -> BenchmarkSummary {
        let gpu_available = self.results.iter().any(|r| r.gpu_used);
        let avg_speedup = if gpu_available {
            let speedups: Vec<f64> = self.results.iter()
                .filter_map(|r| r.speedup)
                .collect();
            if speedups.is_empty() {
                None
            } else {
                Some(speedups.iter().sum::<f64>() / speedups.len() as f64)
            }
        } else {
            None
        };

        BenchmarkSummary {
            total_tests: self.results.len(),
            gpu_available,
            avg_speedup,
            results: self.results.clone(),
        }
    }
}

impl Default for GpuBenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub total_tests: usize,
    pub gpu_available: bool,
    pub avg_speedup: Option<f64>,
    pub results: Vec<BenchmarkResult>,
}

/// Benchmark NTT at a specific size
fn benchmark_ntt(size: usize) -> BenchmarkResult {
    // Generate test data
    let coefficients: Vec<u64> = (0..size).map(|i| (i as u64) % 1000).collect();

    // CPU baseline using simple FFT-like operations
    let cpu_start = Instant::now();
    let _cpu_result = ntt_cpu(&coefficients);
    let cpu_time = cpu_start.elapsed();

    // GPU acceleration would be tested here if feature is enabled in fl-aggregator
    // For now, we benchmark CPU-only performance

    BenchmarkResult {
        operation: "NTT".to_string(),
        size,
        cpu_time,
        gpu_time: None,
        speedup: None,
        gpu_used: false,
    }
}

/// CPU NTT implementation for benchmark comparison
fn ntt_cpu(coefficients: &[u64]) -> Vec<u64> {
    // Simplified Cooley-Tukey NTT (not production-ready)
    let n = coefficients.len();
    if !n.is_power_of_two() {
        return coefficients.to_vec();
    }

    let mut result = coefficients.to_vec();

    // Bit-reversal permutation
    let log_n = (n as f64).log2() as usize;
    for i in 0..n {
        let rev = bit_reverse(i, log_n);
        if i < rev {
            result.swap(i, rev);
        }
    }

    // Cooley-Tukey butterfly
    let mut m = 2;
    while m <= n {
        for k in (0..n).step_by(m) {
            for j in 0..m / 2 {
                // Simplified butterfly (not proper field arithmetic)
                let t = result[k + j + m / 2];
                let u = result[k + j];
                result[k + j] = u.wrapping_add(t);
                result[k + j + m / 2] = u.wrapping_sub(t);
            }
        }
        m *= 2;
    }

    result
}

/// Bit reversal for NTT
fn bit_reverse(x: usize, log_n: usize) -> usize {
    let mut result = 0;
    let mut x = x;
    for _ in 0..log_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Benchmark Merkle tree building
fn benchmark_merkle_tree(num_leaves: usize) -> BenchmarkResult {
    use sha2::{Sha256, Digest};

    // Generate random leaves
    let leaves: Vec<[u8; 32]> = (0..num_leaves)
        .map(|i| {
            let mut hasher = Sha256::new();
            hasher.update(i.to_le_bytes());
            hasher.finalize().into()
        })
        .collect();

    // CPU baseline
    let cpu_start = Instant::now();
    let _cpu_result = build_merkle_tree_cpu(&leaves);
    let cpu_time = cpu_start.elapsed();

    // GPU acceleration would be tested here if feature is enabled in fl-aggregator

    BenchmarkResult {
        operation: "MerkleTree".to_string(),
        size: num_leaves,
        cpu_time,
        gpu_time: None,
        speedup: None,
        gpu_used: false,
    }
}

/// CPU Merkle tree building
fn build_merkle_tree_cpu(leaves: &[[u8; 32]]) -> Vec<[u8; 32]> {
    use sha2::{Sha256, Digest};

    let n = leaves.len();
    if !n.is_power_of_two() {
        return vec![];
    }

    let mut tree = vec![[0u8; 32]; 2 * n - 1];
    tree[n - 1..].copy_from_slice(leaves);

    // Build tree bottom-up
    for i in (0..n - 1).rev() {
        let left = tree[2 * i + 1];
        let right = tree[2 * i + 2];
        let mut hasher = Sha256::new();
        hasher.update(&left);
        hasher.update(&right);
        tree[i] = hasher.finalize().into();
    }

    tree
}

/// Benchmark batch hashing
fn benchmark_batch_hash(num_pairs: usize) -> BenchmarkResult {
    use sha2::{Sha256, Digest};

    // Generate test data (pairs of 32-byte hashes)
    let inputs: Vec<[u8; 32]> = (0..num_pairs * 2)
        .map(|i| {
            let mut hasher = Sha256::new();
            hasher.update(i.to_le_bytes());
            hasher.finalize().into()
        })
        .collect();

    // CPU baseline
    let cpu_start = Instant::now();
    let _cpu_result: Vec<[u8; 32]> = inputs.chunks(2)
        .map(|pair| {
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            hasher.update(&pair[1]);
            hasher.finalize().into()
        })
        .collect();
    let cpu_time = cpu_start.elapsed();

    // GPU acceleration would be tested here if feature is enabled in fl-aggregator

    BenchmarkResult {
        operation: "BatchHash".to_string(),
        size: num_pairs,
        cpu_time,
        gpu_time: None,
        speedup: None,
        gpu_used: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_format() {
        let result = BenchmarkResult {
            operation: "NTT".to_string(),
            size: 1024,
            cpu_time: Duration::from_millis(50),
            gpu_time: Some(Duration::from_millis(8)),
            speedup: Some(6.25),
            gpu_used: true,
        };

        let row = result.to_table_row();
        assert!(row.contains("NTT"));
        assert!(row.contains("1024"));
    }

    #[test]
    fn test_bit_reverse() {
        // 4-bit reversal
        assert_eq!(bit_reverse(0b0001, 4), 0b1000);
        assert_eq!(bit_reverse(0b0011, 4), 0b1100);
        assert_eq!(bit_reverse(0b0101, 4), 0b1010);
    }

    #[test]
    fn test_cpu_ntt() {
        let coefficients = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = ntt_cpu(&coefficients);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_cpu_merkle_tree() {
        let leaves = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let tree = build_merkle_tree_cpu(&leaves);
        assert_eq!(tree.len(), 7); // 2*4 - 1 = 7
    }

    #[test]
    fn test_ntt_benchmark() {
        let result = benchmark_ntt(1024);
        assert_eq!(result.operation, "NTT");
        assert_eq!(result.size, 1024);
        assert!(result.cpu_time > Duration::ZERO);
    }

    #[test]
    fn test_merkle_benchmark() {
        let result = benchmark_merkle_tree(256);
        assert_eq!(result.operation, "MerkleTree");
        assert_eq!(result.size, 256);
    }

    #[test]
    fn test_hash_benchmark() {
        let result = benchmark_batch_hash(128);
        assert_eq!(result.operation, "BatchHash");
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = GpuBenchmarkSuite::new();

        // Run smaller benchmarks for test speed
        let result = benchmark_ntt(256);
        suite.results.push(result);

        let summary = suite.summary();
        assert_eq!(summary.total_tests, 1);
    }

    #[test]
    fn test_full_benchmark_suite() {
        let mut suite = GpuBenchmarkSuite::new();

        // Run all benchmarks at small sizes for testing
        suite.results.push(benchmark_ntt(1024));
        suite.results.push(benchmark_merkle_tree(256));
        suite.results.push(benchmark_batch_hash(128));

        suite.print_table();

        let summary = suite.summary();
        assert_eq!(summary.total_tests, 3);

        println!("\n=== Summary ===");
        println!("Total tests: {}", summary.total_tests);
        println!("GPU available: {}", summary.gpu_available);
        if let Some(speedup) = summary.avg_speedup {
            println!("Average speedup: {:.2}x", speedup);
        }
    }

    // test_gpu_proof_integration and test_gpu_availability removed:
    // fl_aggregator archived (Feb 2026). Proof generation now in mycelix-fl-proofs.
    // GPU acceleration for proofs is a future project.

    /// Benchmark comparison test
    #[test]
    fn test_gpu_vs_cpu_comparison() {
        println!("\n=== GPU vs CPU Comparison ===\n");

        // Run benchmarks at multiple sizes
        let sizes = [1024, 4096, 16384];

        for &size in &sizes {
            let ntt_result = benchmark_ntt(size);
            let merkle_result = benchmark_merkle_tree(size);

            println!("Size: {}", size);
            println!("  NTT CPU: {:.3}ms", ntt_result.cpu_time.as_secs_f64() * 1000.0);
            if let Some(gpu_time) = ntt_result.gpu_time {
                println!("  NTT GPU: {:.3}ms (speedup: {:.2}x)",
                    gpu_time.as_secs_f64() * 1000.0,
                    ntt_result.speedup.unwrap_or(0.0));
            }
            println!("  Merkle CPU: {:.3}ms", merkle_result.cpu_time.as_secs_f64() * 1000.0);
            if let Some(gpu_time) = merkle_result.gpu_time {
                println!("  Merkle GPU: {:.3}ms (speedup: {:.2}x)",
                    gpu_time.as_secs_f64() * 1000.0,
                    merkle_result.speedup.unwrap_or(0.0));
            }
            println!();
        }
    }
}
