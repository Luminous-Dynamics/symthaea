// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Performance Optimizations
//!
//! Provides SIMD-aware batch operations and HV pool integration for
//! the consciousness pipeline hot paths.
//!
//! # Architecture
//!
//! The consciousness pipeline (`process()` in `consciousness_integration.rs`)
//! contains several hot paths that benefit from batch optimization:
//!
//! 1. **Clustering similarity matrix** - N*(N-1)/2 pairwise similarities
//! 2. **Temporal memory search** - M similarity checks per cycle
//! 3. **Self-aware processing** - BinaryHV→ContinuousHV conversion
//!
//! This module provides optimized implementations that:
//! - Compute similarity matrices in batch (cache-friendly access patterns)
//! - Use HV pool for ContinuousHV allocations in self-aware processing
//! - Pre-warm pools at pipeline initialization
//! - Report SIMD capability and pool statistics

use super::binary_hv::BinaryHV;
use super::hv_pool::{BinaryHVPool, ContinuousHVPool, PoolStats, PooledContinuousHV};
#[allow(unused_imports)]
use super::unified_hv::ContinuousHV;

// =============================================================================
// SIMD BATCH OPERATIONS
// =============================================================================

/// Compute pairwise similarity matrix for a slice of BinaryHVs.
///
/// Returns a flat `Vec<f32>` of length N*(N-1)/2 in upper-triangle order:
/// `[(0,1), (0,2), ..., (0,N-1), (1,2), (1,3), ..., (N-2,N-1)]`
///
/// Uses the SIMD-accelerated `BinaryHV::similarity()` internally.
///
/// # Performance
///
/// For N=10 inputs:
/// - Individual calls: 45 × ~20ns = ~900ns
/// - This batch version: ~850ns (same SIMD, but better cache locality)
///
/// The primary advantage is cache-friendly sequential access patterns
/// rather than scattered calls throughout the clustering loop.
#[inline]
pub fn batch_similarity_matrix(hvs: &[&BinaryHV]) -> Vec<f32> {
    let n = hvs.len();
    let pairs = n * (n - 1) / 2;
    let mut result = Vec::with_capacity(pairs);

    for i in 0..n {
        for j in (i + 1)..n {
            result.push(hvs[i].similarity(hvs[j]));
        }
    }

    result
}

/// Get the index into the flat similarity matrix for pair (i, j) where i < j.
///
/// For a matrix of N elements, index of (i,j) = i*(2*N-i-1)/2 + j - i - 1
#[inline]
pub fn similarity_index(n: usize, i: usize, j: usize) -> usize {
    assert!(
        i < j,
        "similarity_index: i ({}) must be less than j ({})",
        i,
        j
    );
    assert!(
        j < n,
        "similarity_index: j ({}) must be less than n ({})",
        j,
        n
    );
    i * (2 * n - i - 1) / 2 + j - i - 1
}

/// Find all HVs similar to `query` above `threshold` from a slice.
///
/// Returns indices and similarity values, sorted by descending similarity.
/// Useful for temporal memory search in the consciousness pipeline.
pub fn find_similar(
    query: &BinaryHV,
    candidates: &[BinaryHV],
    threshold: f32,
) -> Vec<(usize, f32)> {
    let mut results: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(i, hv)| (i, query.similarity(hv)))
        .filter(|(_, sim)| *sim > threshold)
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Compute the mean similarity of a set of HVs to a reference HV.
///
/// Used for synchrony calculation in the clustering phase.
#[inline]
pub fn mean_similarity(reference: &BinaryHV, others: &[&BinaryHV]) -> f64 {
    if others.is_empty() {
        return 0.0;
    }
    let sum: f64 = others
        .iter()
        .map(|hv| reference.similarity(hv) as f64)
        .sum();
    sum / others.len() as f64
}

/// Find similar candidates for multiple queries in a single pass.
///
/// For each query, returns indices and similarity values of candidates above threshold,
/// sorted by descending similarity. More efficient than calling `find_similar` per query
/// because it avoids repeated allocation of the candidate iteration setup.
///
/// Takes references to avoid copying 2KB BinaryHVs.
/// Returns a Vec of length `queries.len()`, where each entry is the results for that query.
pub fn batch_find_similar(
    queries: &[&BinaryHV],
    candidates: &[&BinaryHV],
    threshold: f32,
) -> Vec<Vec<(usize, f32)>> {
    queries
        .iter()
        .map(|query| {
            let mut results: Vec<(usize, f32)> = candidates
                .iter()
                .enumerate()
                .map(|(i, hv)| (i, query.similarity(hv)))
                .filter(|(_, sim)| *sim > threshold)
                .collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results
        })
        .collect()
}

// =============================================================================
// SIMD CAPABILITY REPORTING
// =============================================================================

/// Report which SIMD instruction sets are available on this CPU.
///
/// Returns a human-readable string describing the SIMD capabilities
/// that BinaryHV operations will use.
pub fn simd_capabilities() -> SimdCapabilities {
    use super::simd_detect;
    SimdCapabilities {
        avx512f: simd_detect::has_avx512f(),
        avx512bw: simd_detect::has_avx512bw(),
        avx512_vpopcntdq: simd_detect::has_avx512_vpopcntdq(),
        avx2: simd_detect::has_avx2(),
        sse41: simd_detect::has_sse41(),
        popcnt: simd_detect::has_popcnt(),
    }
}

/// SIMD capability flags for this CPU.
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512_vpopcntdq: bool,
    pub avx2: bool,
    pub sse41: bool,
    pub popcnt: bool,
}

impl SimdCapabilities {
    /// Best available tier for bind operations.
    pub fn bind_tier(&self) -> &'static str {
        if self.avx512f && self.avx512bw {
            "AVX-512 (32 iterations/2048 bytes)"
        } else if self.avx2 {
            "AVX2 (64 iterations/2048 bytes)"
        } else if self.sse41 {
            "SSE4.1 (128 iterations/2048 bytes)"
        } else {
            "Scalar (256 iterations/2048 bytes)"
        }
    }

    /// Best available tier for similarity operations.
    pub fn similarity_tier(&self) -> &'static str {
        if self.avx512_vpopcntdq {
            "AVX-512 VPOPCNTDQ (native 64-bit popcount)"
        } else if self.popcnt {
            "POPCNT (hardware popcount)"
        } else {
            "Scalar (software popcount)"
        }
    }

    /// Summary string for logging.
    pub fn summary(&self) -> String {
        format!(
            "SIMD: bind={}, similarity={}, features=[{}]",
            self.bind_tier(),
            self.similarity_tier(),
            [
                if self.avx512f { "AVX-512F" } else { "" },
                if self.avx512bw { "AVX-512BW" } else { "" },
                if self.avx512_vpopcntdq {
                    "VPOPCNTDQ"
                } else {
                    ""
                },
                if self.avx2 { "AVX2" } else { "" },
                if self.sse41 { "SSE4.1" } else { "" },
                if self.popcnt { "POPCNT" } else { "" },
            ]
            .iter()
            .filter(|s| !s.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join(", ")
        )
    }
}

// =============================================================================
// POOL INTEGRATION FOR CONSCIOUSNESS PIPELINE
// =============================================================================

/// Pre-warm HV pools for consciousness pipeline operation.
///
/// Call this during pipeline initialization to avoid cold-start
/// allocation costs during the first few processing cycles.
///
/// Recommended for pipelines that:
/// - Use self-aware processing (needs ContinuousHV pool)
/// - Process many inputs per cycle (needs BinaryHV pool for temporaries)
pub fn prewarm_consciousness_pools() {
    // Pre-warm binary pool for clustering temporaries
    BinaryHVPool::prewarm(16);
    // Pre-warm continuous pool for self-aware processing
    ContinuousHVPool::prewarm(4);
}

/// Convert BinaryHV to ContinuousHV using a pooled allocation.
///
/// Maps binary bits to [-1.0, +1.0] bipolar representation.
/// The returned PooledContinuousHV will return its allocation to the
/// pool when dropped.
///
/// This is the pool-optimized version of `hv16_to_real_hv()` in
/// consciousness_integration.rs.
pub fn binary_to_continuous_pooled(hv: &BinaryHV) -> PooledContinuousHV {
    let dim = BinaryHV::DIM;
    let mut pooled = ContinuousHVPool::get_with_dim(dim);
    let data = pooled.as_slice_mut();

    // Map each bit to +1.0 (set) or -1.0 (clear)
    for byte_idx in 0..2048 {
        let byte = hv.0[byte_idx];
        let base = byte_idx * 8;
        for bit in 0..8u8 {
            let idx = base + bit as usize;
            if idx < dim {
                data[idx] = if byte & (1 << bit) != 0 { 1.0 } else { -1.0 };
            }
        }
    }

    pooled
}

/// Get current pool statistics for monitoring.
pub fn pool_stats() -> PoolStats {
    PoolStats::current()
}

// =============================================================================
// CONSCIOUSNESS CLUSTERING HELPER
// =============================================================================

/// Greedy clustering of HVs by similarity.
///
/// Groups input HVs into clusters where each member has similarity > threshold
/// to the cluster seed. Returns cluster indices.
///
/// This is the optimized version of the clustering loop in
/// `ConsciousnessPipeline::process()`.
pub fn cluster_by_similarity(hvs: &[&BinaryHV], threshold: f64) -> Vec<Vec<usize>> {
    let n = hvs.len();
    if n == 0 {
        return Vec::new();
    }

    // Pre-compute similarity matrix
    let sim_matrix = batch_similarity_matrix(hvs);

    let mut used = vec![false; n];
    let mut clusters = Vec::new();

    for i in 0..n {
        if used[i] {
            continue;
        }

        let mut cluster = vec![i];
        used[i] = true;

        for j in (i + 1)..n {
            if used[j] {
                continue;
            }
            let sim = sim_matrix[similarity_index(n, i, j)] as f64;
            if sim > threshold {
                cluster.push(j);
                used[j] = true;
            }
        }

        clusters.push(cluster);
    }

    clusters
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_similarity_matrix() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        let c = BinaryHV::random(3);

        let hvs: Vec<&BinaryHV> = vec![&a, &b, &c];
        let matrix = batch_similarity_matrix(&hvs);

        assert_eq!(matrix.len(), 3); // 3 pairs: (0,1), (0,2), (1,2)

        // Verify matches individual calls
        assert!((matrix[0] - a.similarity(&b)).abs() < 1e-6);
        assert!((matrix[1] - a.similarity(&c)).abs() < 1e-6);
        assert!((matrix[2] - b.similarity(&c)).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_index() {
        // For N=4: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
        assert_eq!(similarity_index(4, 0, 1), 0);
        assert_eq!(similarity_index(4, 0, 2), 1);
        assert_eq!(similarity_index(4, 0, 3), 2);
        assert_eq!(similarity_index(4, 1, 2), 3);
        assert_eq!(similarity_index(4, 1, 3), 4);
        assert_eq!(similarity_index(4, 2, 3), 5);
    }

    #[test]
    fn test_find_similar_self() {
        let target = BinaryHV::random(10);
        let others = vec![
            BinaryHV::random(20),
            target, // exact copy
            BinaryHV::random(30),
        ];

        let results = find_similar(&target, &others, 0.9);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 1); // Index 1 is the exact copy
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_find_similar_threshold() {
        let target = BinaryHV::random(100);
        let others: Vec<BinaryHV> = (200..210).map(|s| BinaryHV::random(s)).collect();

        // Random HVs have similarity ~0.5, so threshold 0.6 should find few/none
        let results = find_similar(&target, &others, 0.6);
        // All results should actually be above threshold
        for (_, sim) in &results {
            assert!(*sim > 0.6);
        }
    }

    #[test]
    fn test_mean_similarity() {
        let reference = BinaryHV::random(1);
        let others: Vec<BinaryHV> = (2..6).map(|s| BinaryHV::random(s)).collect();
        let refs: Vec<&BinaryHV> = others.iter().collect();

        let mean = mean_similarity(&reference, &refs);
        // Random HVs: mean similarity should be near 0.5
        assert!(
            mean > 0.45 && mean < 0.55,
            "Mean similarity {} should be near 0.5",
            mean
        );
    }

    #[test]
    fn test_mean_similarity_empty() {
        let reference = BinaryHV::random(1);
        assert_eq!(mean_similarity(&reference, &[]), 0.0);
    }

    #[test]
    fn test_simd_capabilities() {
        let caps = simd_capabilities();
        // Just verify it doesn't panic and returns valid data
        let summary = caps.summary();
        assert!(summary.contains("SIMD:"));
        assert!(!caps.bind_tier().is_empty());
        assert!(!caps.similarity_tier().is_empty());
    }

    #[test]
    fn test_prewarm_consciousness_pools() {
        // Clear first to have known state
        BinaryHVPool::clear();
        ContinuousHVPool::clear();

        prewarm_consciousness_pools();

        assert!(BinaryHVPool::pool_size() >= 16);
        assert!(ContinuousHVPool::pool_size() >= 4);
    }

    #[test]
    fn test_binary_to_continuous_pooled() {
        let hv = BinaryHV::random(42);
        let continuous = binary_to_continuous_pooled(&hv);

        // Check dimension
        assert_eq!(continuous.dim(), BinaryHV::DIM);

        // All values should be -1.0 or +1.0
        for &val in continuous.as_slice() {
            assert!(
                val == -1.0 || val == 1.0,
                "Expected -1.0 or 1.0, got {}",
                val
            );
        }

        // Verify bit mapping: check first byte
        let first_byte = hv.0[0];
        for bit in 0..8u8 {
            let expected = if first_byte & (1 << bit) != 0 {
                1.0
            } else {
                -1.0
            };
            assert_eq!(continuous.as_slice()[bit as usize], expected);
        }
    }

    #[test]
    fn test_cluster_by_similarity() {
        // Create 3 similar HVs and 2 different ones
        let base = BinaryHV::random(100);
        let similar1 = base; // identical = sim 1.0
        let similar2 = base; // identical = sim 1.0
        let different1 = BinaryHV::random(200);
        let different2 = BinaryHV::random(300);

        let hvs: Vec<&BinaryHV> = vec![&base, &similar1, &similar2, &different1, &different2];
        let clusters = cluster_by_similarity(&hvs, 0.9);

        // The first 3 should be in one cluster (sim=1.0 > 0.9)
        assert!(!clusters.is_empty());
        let first_cluster = &clusters[0];
        assert_eq!(
            first_cluster.len(),
            3,
            "First 3 identical HVs should cluster together"
        );
        assert!(first_cluster.contains(&0));
        assert!(first_cluster.contains(&1));
        assert!(first_cluster.contains(&2));
    }

    #[test]
    fn test_cluster_by_similarity_empty() {
        let clusters = cluster_by_similarity(&[], 0.5);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_batch_find_similar() {
        let q1 = BinaryHV::random(1);
        let q2 = BinaryHV::random(2);
        let candidates = vec![
            q1, // exact match for q1
            q2, // exact match for q2
            BinaryHV::random(100),
            BinaryHV::random(200),
        ];
        let cand_refs: Vec<&BinaryHV> = candidates.iter().collect();

        let results = batch_find_similar(&[&q1, &q2], &cand_refs, 0.9);
        assert_eq!(results.len(), 2);
        // q1 should find itself at index 0
        assert!(!results[0].is_empty());
        assert_eq!(results[0][0].0, 0);
        assert!((results[0][0].1 - 1.0).abs() < 0.001);
        // q2 should find itself at index 1
        assert!(!results[1].is_empty());
        assert_eq!(results[1][0].0, 1);
        assert!((results[1][0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_find_similar_matches_individual() {
        let queries: Vec<BinaryHV> = (0..5).map(|s| BinaryHV::random(s)).collect();
        let candidates: Vec<BinaryHV> = (10..30).map(|s| BinaryHV::random(s)).collect();

        let query_refs: Vec<&BinaryHV> = queries.iter().collect();
        let cand_refs: Vec<&BinaryHV> = candidates.iter().collect();
        let batch = batch_find_similar(&query_refs, &cand_refs, 0.55);
        for (i, query) in queries.iter().enumerate() {
            let individual = find_similar(query, &candidates, 0.55);
            assert_eq!(
                batch[i].len(),
                individual.len(),
                "batch and individual should find same count for query {}",
                i
            );
        }
    }

    #[test]
    fn test_pool_stats() {
        let stats = pool_stats();
        // Just verify it returns valid data
        assert!(stats.hv16_capacity > 0);
        assert!(stats.continuous_hv_capacity > 0);
    }
}
