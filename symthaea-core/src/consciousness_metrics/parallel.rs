// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Parallel Entropy Computation
//!
//! Uses rayon for parallel processing of entropy calculations.

use crate::hdc::unified_hv::ContinuousHV;

use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::RwLock;

use super::{ContinuousEntropyEstimator, TruePartition, TruePhiCalculator, TruePhiResult};

/// Thread-safe cache for expensive computations
pub(crate) struct EntropyCache {
    /// Cache for entropy values (key: hash of HV values)
    pub(crate) entropy_cache: RwLock<HashMap<u64, f64>>,
    /// Cache for MI values (key: hash of both HVs)
    pub(crate) mi_cache: RwLock<HashMap<(u64, u64), f64>>,
    /// Maximum cache size
    max_size: usize,
}

impl EntropyCache {
    fn new(max_size: usize) -> Self {
        Self {
            entropy_cache: RwLock::new(HashMap::with_capacity(max_size / 2)),
            mi_cache: RwLock::new(HashMap::with_capacity(max_size / 2)),
            max_size,
        }
    }

    /// Compute a fast hash for a ContinuousHV
    fn hash_hv(hv: &ContinuousHV) -> u64 {
        // Sample a few values for fast hashing
        let mut hash = 0u64;
        let step = hv.values.len().max(1) / 16;
        for i in (0..hv.values.len()).step_by(step.max(1)) {
            let bits = hv.values[i].to_bits() as u64;
            hash = hash.wrapping_mul(31).wrapping_add(bits);
        }
        hash
    }

    pub(crate) fn get_entropy(&self, hv: &ContinuousHV) -> Option<f64> {
        let hash = Self::hash_hv(hv);
        self.entropy_cache.read().ok()?.get(&hash).copied()
    }

    pub(crate) fn set_entropy(&self, hv: &ContinuousHV, value: f64) {
        let hash = Self::hash_hv(hv);
        if let Ok(mut cache) = self.entropy_cache.write() {
            if cache.len() >= self.max_size / 2 {
                // Simple eviction: clear half the cache
                let keys: Vec<_> = cache.keys().take(cache.len() / 2).copied().collect();
                for k in keys {
                    cache.remove(&k);
                }
            }
            cache.insert(hash, value);
        }
    }

    pub(crate) fn get_mi(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> Option<f64> {
        let hash1 = Self::hash_hv(hv1);
        let hash2 = Self::hash_hv(hv2);
        let key = if hash1 <= hash2 {
            (hash1, hash2)
        } else {
            (hash2, hash1)
        };
        self.mi_cache.read().ok()?.get(&key).copied()
    }

    pub(crate) fn set_mi(&self, hv1: &ContinuousHV, hv2: &ContinuousHV, value: f64) {
        let hash1 = Self::hash_hv(hv1);
        let hash2 = Self::hash_hv(hv2);
        let key = if hash1 <= hash2 {
            (hash1, hash2)
        } else {
            (hash2, hash1)
        };
        if let Ok(mut cache) = self.mi_cache.write() {
            if cache.len() >= self.max_size / 2 {
                let keys: Vec<_> = cache.keys().take(cache.len() / 2).copied().collect();
                for k in keys {
                    cache.remove(&k);
                }
            }
            cache.insert(key, value);
        }
    }
}

/// Global entropy cache
pub(crate) static ENTROPY_CACHE: Lazy<EntropyCache> = Lazy::new(|| EntropyCache::new(10000));

/// Parallel entropy calculator using rayon
///
/// Provides parallel versions of entropy computations for batch processing.
/// Uses rayon's work-stealing thread pool for efficient parallelization.
#[derive(Debug, Clone)]
pub struct ParallelEntropyCalculator {
    /// Base estimator for entropy computation
    estimator: ContinuousEntropyEstimator,
    /// Whether to use caching
    use_cache: bool,
}

impl Default for ParallelEntropyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelEntropyCalculator {
    /// Create a new parallel calculator with default estimator
    pub fn new() -> Self {
        Self {
            estimator: ContinuousEntropyEstimator::fast(),
            use_cache: true,
        }
    }

    /// Create with custom estimator
    pub fn with_estimator(estimator: ContinuousEntropyEstimator) -> Self {
        Self {
            estimator,
            use_cache: true,
        }
    }

    /// Create without caching (for benchmarking)
    pub fn without_cache() -> Self {
        Self {
            estimator: ContinuousEntropyEstimator::fast(),
            use_cache: false,
        }
    }

    /// Compute entropy for multiple vectors in parallel
    ///
    /// Returns entropies in the same order as input vectors.
    pub fn entropy_batch(&self, vectors: &[ContinuousHV]) -> Vec<f64> {
        if self.use_cache {
            vectors
                .par_iter()
                .map(|hv| {
                    if let Some(cached) = ENTROPY_CACHE.get_entropy(hv) {
                        cached
                    } else {
                        let h = self.estimator.entropy(hv);
                        ENTROPY_CACHE.set_entropy(hv, h);
                        h
                    }
                })
                .collect()
        } else {
            vectors
                .par_iter()
                .map(|hv| self.estimator.entropy(hv))
                .collect()
        }
    }

    /// Compute mutual information matrix in parallel
    ///
    /// Returns symmetric matrix where M[i][j] = I(X_i; X_j).
    /// Diagonal contains entropies H(X_i).
    pub fn mutual_information_matrix(&self, vectors: &[ContinuousHV]) -> Vec<Vec<f64>> {
        let n = vectors.len();
        if n == 0 {
            return vec![];
        }

        // First compute all entropies in parallel
        let entropies = self.entropy_batch(vectors);

        // Create pair indices for parallel processing
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();

        // Compute all pairwise MIs in parallel
        let mis: Vec<((usize, usize), f64)> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let mi = if self.use_cache {
                    if let Some(cached) = ENTROPY_CACHE.get_mi(&vectors[i], &vectors[j]) {
                        cached
                    } else {
                        let mi = self
                            .estimator
                            .mutual_information_fast(&vectors[i], &vectors[j]);
                        ENTROPY_CACHE.set_mi(&vectors[i], &vectors[j], mi);
                        mi
                    }
                } else {
                    self.estimator
                        .mutual_information_fast(&vectors[i], &vectors[j])
                };
                ((i, j), mi)
            })
            .collect();

        // Build matrix
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = entropies[i];
        }
        for ((i, j), mi) in mis {
            matrix[i][j] = mi;
            matrix[j][i] = mi;
        }

        matrix
    }

    /// Compute effective information in parallel
    ///
    /// EI = Σ_{i<j} I(X_i; X_j)
    pub fn effective_information(&self, vectors: &[ContinuousHV]) -> f64 {
        let n = vectors.len();
        if n < 2 {
            return 0.0;
        }

        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();

        pairs
            .par_iter()
            .map(|&(i, j)| {
                if self.use_cache {
                    if let Some(cached) = ENTROPY_CACHE.get_mi(&vectors[i], &vectors[j]) {
                        cached
                    } else {
                        let mi = self
                            .estimator
                            .mutual_information_fast(&vectors[i], &vectors[j]);
                        ENTROPY_CACHE.set_mi(&vectors[i], &vectors[j], mi);
                        mi
                    }
                } else {
                    self.estimator
                        .mutual_information_fast(&vectors[i], &vectors[j])
                }
            })
            .sum()
    }

    /// Parallel true Φ computation
    ///
    /// Uses parallel MI matrix construction for faster MIP search.
    pub fn compute_true_phi_parallel(&self, components: &[ContinuousHV]) -> TruePhiResult {
        let n = components.len();

        if n < 2 {
            return TruePhiResult {
                phi: 0.0,
                system_ei: 0.0,
                mip_ei: 0.0,
                mip: TruePartition {
                    part_a: (0..n).collect(),
                    part_b: vec![],
                },
                component_entropies: if n == 1 {
                    vec![self.estimator.entropy(&components[0])]
                } else {
                    vec![]
                },
                mutual_information_matrix: vec![],
            };
        }

        // Build MI matrix in parallel
        let mi_matrix = self.mutual_information_matrix(components);

        // Extract component entropies (diagonal)
        let component_entropies: Vec<f64> = (0..n).map(|i| mi_matrix[i][i]).collect();

        // Compute system EI from matrix (sum of upper triangle)
        let mut system_ei = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                system_ei += mi_matrix[i][j];
            }
        }

        // Find MIP using existing algorithm
        let calc = TruePhiCalculator::new();
        let (mip, mip_ei) = calc.find_true_mip(components);

        let phi = (system_ei - mip_ei).max(0.0);

        TruePhiResult {
            phi,
            system_ei,
            mip_ei,
            mip,
            component_entropies,
            mutual_information_matrix: mi_matrix,
        }
    }

    /// Clear the entropy cache
    pub fn clear_cache() {
        if let Ok(mut cache) = ENTROPY_CACHE.entropy_cache.write() {
            cache.clear();
        }
        if let Ok(mut cache) = ENTROPY_CACHE.mi_cache.write() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn cache_stats() -> (usize, usize) {
        let entropy_size = ENTROPY_CACHE
            .entropy_cache
            .read()
            .map(|c| c.len())
            .unwrap_or(0);
        let mi_size = ENTROPY_CACHE.mi_cache.read().map(|c| c.len()).unwrap_or(0);
        (entropy_size, mi_size)
    }
}

/// Cached entropy calculator for single-threaded use
///
/// Wraps an estimator with caching for repeated computations.
#[derive(Debug, Clone)]
pub struct CachedEntropyCalculator {
    estimator: ContinuousEntropyEstimator,
}

impl Default for CachedEntropyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl CachedEntropyCalculator {
    /// Create a new cached calculator
    pub fn new() -> Self {
        Self {
            estimator: ContinuousEntropyEstimator::fast(),
        }
    }

    /// Create with custom estimator
    pub fn with_estimator(estimator: ContinuousEntropyEstimator) -> Self {
        Self { estimator }
    }

    /// Compute entropy with caching
    pub fn entropy(&self, hv: &ContinuousHV) -> f64 {
        if let Some(cached) = ENTROPY_CACHE.get_entropy(hv) {
            return cached;
        }
        let h = self.estimator.entropy(hv);
        ENTROPY_CACHE.set_entropy(hv, h);
        h
    }

    /// Compute mutual information with caching
    pub fn mutual_information(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> f64 {
        if let Some(cached) = ENTROPY_CACHE.get_mi(hv1, hv2) {
            return cached;
        }
        let mi = self.estimator.mutual_information_fast(hv1, hv2);
        ENTROPY_CACHE.set_mi(hv1, hv2, mi);
        mi
    }
}
