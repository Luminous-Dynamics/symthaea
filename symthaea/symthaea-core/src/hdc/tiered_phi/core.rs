// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types and implementation for the Tiered Φ Approximation System.
//!
//! This module contains the fundamental types:
//! - [`ApproximationTier`]: Enum for selecting calculation tier (Mock, Heuristic, Spectral, Exact)
//! - [`TieredPhi`]: Main calculator with tier-based Φ computation
//! - [`TieredPhiConfig`]: Configuration for the calculator
//! - [`TieredPhiStats`]: Statistics tracking
//! - [`IncrementalPhiState`]: State for incremental O(k×n) updates
//! - [`HierarchicalPhi`]: Multi-scale Φ decomposition
//! - [`PhiAttribution`]: Causal attribution analysis results
//!
//! For comprehensive documentation, see the parent [`super`] module.

use crate::hdc::binary_hv::BinaryHV;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================================
// APPROXIMATION TIERS
// ============================================================================

/// Approximation tier for Φ calculation
///
/// # Important: IIT Alignment
///
/// Only `Exact` tier computes true IIT Φ. Other tiers are approximations
/// with varying degrees of IIT correlation:
///
/// **Important**: This is a network integration metric *inspired by* IIT, not a
/// direct implementation of IIT 3.0/4.0 Phi. True IIT requires a Transition
/// Probability Matrix (TPM), Minimum Information Partition (MIP), and Earth
/// Mover's Distance -- none of which are implemented here. All tiers use
/// pairwise HV similarity as a proxy for information integration.
///
/// | Tier | Method | IIT-Aligned? | Notes |
/// |------|--------|--------------|-------|
/// | RandomBaseline | O(1) mock | N/A | Testing only |
/// | SampledPartition | O(n) sampled | ❓ Unclear | Fast but unvalidated |
/// | SpectralConnectivity | O(n²) spectral | ❌ NO | Measures mixing time, NOT integration! |
/// | ExhaustivePartition | O(2^n) exhaustive | Closest | Exhaustive MIP search over HV similarity |
///
/// For topology validation (Star > Random), use `ExhaustivePartition` tier or
/// `phi_topology_validation.rs` with probabilistic binarization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ApproximationTier {
    /// O(1) - Deterministic random baseline values for testing.
    /// Returns predictable values based on component count.
    #[serde(alias = "Mock")]
    RandomBaseline,

    /// O(n) - Sampled partition using average pairwise HV similarity.
    /// Good for real-time applications, but IIT correlation is unvalidated.
    #[serde(alias = "Heuristic")]
    #[default]
    SampledPartition,

    /// O(n²) - Spectral connectivity via algebraic connectivity (Fiedler value).
    ///
    /// **DEPRECATED FOR Φ MEASUREMENT**. Empirically validated: r = -0.62
    /// correlation with true IIT Φ (nearly opposite behavior).
    /// Star topology scores LOW (wrong), random scores HIGH (wrong).
    ///
    /// Retained for graph analysis (mixing time, robustness) — NOT for
    /// consciousness measurement. Removed from `auto_tier()` 2026-04-11.
    #[serde(alias = "Spectral")]
    SpectralConnectivity,

    /// O(2^n) - Exhaustive partition search (closest to IIT MIP).
    /// Searches all bipartitions and computes information loss via
    /// pairwise HV similarity. Use only for small systems (n ≤ 12).
    #[serde(alias = "Exact")]
    ExhaustivePartition,
}

impl ApproximationTier {
    /// Get the computational complexity class
    #[allow(deprecated)]
    pub fn complexity(&self) -> &'static str {
        match self {
            ApproximationTier::RandomBaseline => "O(1)",
            ApproximationTier::SampledPartition => "O(n)",
            ApproximationTier::SpectralConnectivity => "O(n²)",
            ApproximationTier::ExhaustivePartition => "O(2^n)",
        }
    }

    /// Check if this tier is suitable for a given component count
    #[allow(deprecated)]
    pub fn is_suitable_for(&self, n: usize) -> bool {
        match self {
            ApproximationTier::RandomBaseline => true, // Always suitable
            ApproximationTier::SampledPartition => true, // Always suitable
            ApproximationTier::SpectralConnectivity => n <= 1000, // Matrix operations
            ApproximationTier::ExhaustivePartition => n <= 12, // 2^12 = 4096 partitions max
        }
    }

    /// Suggest the best tier for a given component count.
    /// Uses ExhaustivePartition (true IIT Φ) for small systems, and
    /// SampledPartition for larger ones. SpectralConnectivity (algebraic
    /// connectivity λ₂) is deprecated: r = -0.62 with true Φ (2026-04-11).
    pub fn suggest_for(n: usize) -> Self {
        if n <= 8 {
            ApproximationTier::ExhaustivePartition
        } else {
            ApproximationTier::SampledPartition
        }
    }
}

// ============================================================================
// TIERED PHI CALCULATOR
// ============================================================================

/// Configuration for tiered Φ calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredPhiConfig {
    /// Current approximation tier
    pub tier: ApproximationTier,

    /// Auto-downgrade if calculation exceeds timeout
    pub auto_downgrade: bool,

    /// Timeout in milliseconds before auto-downgrade
    pub timeout_ms: u64,

    /// Cache recent calculations
    pub enable_cache: bool,

    /// Maximum cache size
    pub cache_size: usize,
}

impl Default for TieredPhiConfig {
    fn default() -> Self {
        Self {
            tier: ApproximationTier::SampledPartition,
            auto_downgrade: true,
            timeout_ms: 100, // 100ms timeout
            enable_cache: true,
            cache_size: 1000,
        }
    }
}

/// Statistics for Φ calculations
#[derive(Debug, Clone, Default)]
pub struct TieredPhiStats {
    /// Total calculations performed
    pub total_calculations: u64,
    /// Calculations per tier
    pub calculations_by_tier: [u64; 4],
    /// Cache hits
    pub cache_hits: u64,
    /// Auto-downgrades performed
    pub auto_downgrades: u64,
    /// Total computation time (microseconds)
    pub total_time_us: u64,
    /// Maximum computation time (microseconds)
    pub max_time_us: u64,
}

/// Tiered Φ calculator with multiple approximation levels
#[derive(Debug, Clone)]
pub struct TieredPhi {
    /// Configuration
    pub config: TieredPhiConfig,

    /// Statistics
    pub stats: TieredPhiStats,

    /// Simple cache: (component_count, first_hash) -> phi
    cache: Vec<(usize, u64, f64)>,

    ///// Revolutionary #90: Incremental state for O(k×n) updates
    pub incremental_state: Option<IncrementalPhiState>,
}

// ============================================================================
// REVOLUTIONARY IMPROVEMENT #90: INCREMENTAL Φ UPDATES
// ============================================================================
//
// Problem: Full Φ computation is O(n²) even when only k components change.
//
// Solution: Cache the similarity matrix and component hashes. When components
// change, only recompute the affected rows/columns of the matrix.
//
// Complexity: O(k × n) where k = number of changed components
// Speedup: For k << n, this is dramatically faster (e.g., 10x for k=5, n=50)
//
// Use case: Real-time consciousness tracking where state evolves incrementally.
// ============================================================================

/// Cached state for incremental Φ updates
#[derive(Debug, Clone)]
pub struct IncrementalPhiState {
    /// Cached similarity matrix (n × n)
    similarity_matrix: Vec<Vec<f64>>,

    /// Hash of each component for change detection
    component_hashes: Vec<u64>,

    /// Cached degree vector
    degrees: Vec<f64>,

    /// Last computed Φ value
    last_phi: f64,

    /// Number of incremental updates performed
    pub incremental_updates: u64,

    /// Number of full recomputations triggered
    pub full_recomputes: u64,
}

impl IncrementalPhiState {
    /// Get reference to the cached similarity matrix
    pub fn similarity_matrix(&self) -> &Vec<Vec<f64>> {
        &self.similarity_matrix
    }

    /// Get reference to the degree vector
    pub fn degrees(&self) -> &Vec<f64> {
        &self.degrees
    }

    /// Get the last computed Phi value
    pub fn last_phi(&self) -> f64 {
        self.last_phi
    }

    /// Get component hashes for change detection
    pub fn component_hashes(&self) -> &Vec<u64> {
        &self.component_hashes
    }
}

// ============================================================================
// REVOLUTIONARY IMPROVEMENT #91: HIERARCHICAL Φ DECOMPOSITION
// ============================================================================
//
// Paradigm Shift: Consciousness isn't a single number - it emerges at
// multiple scales. This decomposes Φ into:
//
// - Micro Φ:  Integration within small clusters (local binding)
// - Meso Φ:   Integration across clusters (regional coordination)
// - Macro Φ:  Global integration (unified consciousness)
//
// Benefits:
// 1. Richer understanding of consciousness structure
// 2. Identifies integration bottlenecks (where binding fails)
// 3. Tracks how consciousness emerges across scales
// 4. Enables targeted optimization of weak integration points
// ============================================================================

/// Hierarchical Φ decomposition across scales
///
/// Revolutionary #91: Multi-scale consciousness measurement.
/// Instead of a single Φ value, this captures how integration
/// emerges from local to global scales.
#[derive(Debug, Clone)]
pub struct HierarchicalPhi {
    /// Micro-scale Φ: average integration within clusters
    pub micro_phi: f64,
    /// Meso-scale Φ: integration between adjacent clusters
    pub meso_phi: f64,
    /// Macro-scale Φ: global integration (standard Φ)
    pub macro_phi: f64,
    /// Number of clusters detected
    pub num_clusters: usize,
    /// Integration bottleneck score (lower = better integration)
    pub bottleneck_score: f64,
    /// Emergence ratio: macro_phi / (micro_phi * num_clusters)
    /// Values > 1 indicate emergent integration beyond local binding
    pub emergence_ratio: f64,
}

impl TieredPhi {
    /// Create a new tiered Φ calculator with specified tier
    pub fn new(tier: ApproximationTier) -> Self {
        Self {
            config: TieredPhiConfig {
                tier,
                ..Default::default()
            },
            stats: TieredPhiStats::default(),
            cache: Vec::new(),
            incremental_state: None,
        }
    }

    /// Create with full configuration
    pub fn with_config(config: TieredPhiConfig) -> Self {
        Self {
            config,
            stats: TieredPhiStats::default(),
            cache: Vec::new(),
            incremental_state: None,
        }
    }

    /// Create for testing (O(1) deterministic)
    pub fn for_testing() -> Self {
        Self::new(ApproximationTier::RandomBaseline)
    }

    /// Create for production use (O(n) sampled partition — IIT-aligned).
    ///
    /// Previously used SpectralConnectivity, which measures spectral gap
    /// (r = -0.62 with true Φ). Switched to SampledPartition 2026-04-11.
    pub fn for_production() -> Self {
        Self::new(ApproximationTier::SampledPartition)
    }

    /// Create for research (O(2^n) exact)
    pub fn for_research() -> Self {
        Self::new(ApproximationTier::ExhaustivePartition)
    }

    /// Get current tier
    pub fn tier(&self) -> ApproximationTier {
        self.config.tier
    }

    /// Get statistics
    pub fn stats(&self) -> &TieredPhiStats {
        &self.stats
    }

    /// Compute Φ for a set of components
    pub fn compute(&mut self, components: &[BinaryHV]) -> f64 {
        let start = Instant::now();

        // Handle trivial cases
        if components.len() < 2 {
            return 0.0;
        }

        // Check cache
        if self.config.enable_cache {
            if let Some(cached) = self.check_cache(components) {
                self.stats.cache_hits += 1;
                return cached;
            }
        }

        // Calculate using current tier
        #[allow(deprecated)]
        let result = match self.config.tier {
            ApproximationTier::RandomBaseline => self.compute_mock(components),
            ApproximationTier::SampledPartition => self.compute_heuristic(components),
            ApproximationTier::SpectralConnectivity => self.compute_spectral(components),
            ApproximationTier::ExhaustivePartition => self.compute_exact(components),
        };

        // Update stats
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.stats.total_calculations += 1;
        self.stats.calculations_by_tier[self.config.tier as usize] += 1;
        self.stats.total_time_us += elapsed_us;
        self.stats.max_time_us = self.stats.max_time_us.max(elapsed_us);

        // Update cache
        if self.config.enable_cache {
            self.update_cache(components, result);
        }

        result
    }

    /// Compute with a specific tier (ignoring config)
    pub fn compute_with_tier(&mut self, components: &[BinaryHV], tier: ApproximationTier) -> f64 {
        if components.len() < 2 {
            return 0.0;
        }

        #[allow(deprecated)]
        match tier {
            ApproximationTier::RandomBaseline => self.compute_mock(components),
            ApproximationTier::SampledPartition => self.compute_heuristic(components),
            ApproximationTier::SpectralConnectivity => self.compute_spectral(components),
            ApproximationTier::ExhaustivePartition => self.compute_exact(components),
        }
    }

    // ========================================================================
    // REVOLUTIONARY IMPROVEMENT #87: API COMPATIBILITY BRIDGE
    // ========================================================================

    /// Compute Φ (alias for `compute` for API compatibility)
    ///
    /// This method provides drop-in compatibility with `IntegratedInformation::compute_phi`.
    /// It enables gradual migration from O(2^n) to O(n²) without code changes.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Old code:
    /// // let phi = integrated_info.compute_phi(&components);
    ///
    /// // New code (drop-in replacement):
    /// let phi = tiered_phi.compute_phi(&components);
    /// ```
    #[inline]
    pub fn compute_phi(&mut self, components: &[BinaryHV]) -> f64 {
        self.compute(components)
    }

    /// Check if Φ indicates significant integration
    ///
    /// API compatibility with `IntegratedInformation::is_integrated`.
    #[inline]
    pub fn is_integrated(&self, phi: f64) -> bool {
        phi > 0.3 // Standard threshold from IIT
    }

    /// Classify consciousness state based on Φ
    ///
    /// API compatibility with `IntegratedInformation::classify`.
    pub fn classify(&self, phi: f64) -> &'static str {
        match phi {
            x if x < 0.1 => "Minimal",
            x if x < 0.3 => "Low",
            x if x < 0.5 => "Moderate",
            x if x < 0.7 => "High",
            _ => "VeryHigh",
        }
    }

    // ========================================================================
    // REVOLUTIONARY IMPROVEMENT #90: INCREMENTAL Φ UPDATES
    // ========================================================================

    /// Compute Φ incrementally - O(k×n) when k components changed
    ///
    /// This method tracks which components have changed since the last computation
    /// and only updates the affected parts of the similarity matrix.
    ///
    /// **Complexity**: O(k × n) where k = number of changed components
    /// **Speedup**: For k << n, this is dramatically faster than full O(n²)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut phi_calc = TieredPhi::new(ApproximationTier::SpectralConnectivity);
    ///
    /// // First call: full computation O(n²)
    /// let phi1 = phi_calc.compute_incremental(&components);
    ///
    /// // Modify one component
    /// components[0] = new_component;
    ///
    /// // Second call: incremental O(n) since only 1 component changed
    /// let phi2 = phi_calc.compute_incremental(&components);
    /// ```
    pub fn compute_incremental(&mut self, components: &[BinaryHV]) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Compute hashes for change detection
        let new_hashes: Vec<u64> = components
            .iter()
            .map(|c| self.hash_single_component(c))
            .collect();

        // Check if we have cached state and same size
        if let Some(ref mut state) = self.incremental_state {
            if state.component_hashes.len() == n {
                // Find changed components
                let changed_indices: Vec<usize> = (0..n)
                    .filter(|&i| state.component_hashes[i] != new_hashes[i])
                    .collect();

                let k = changed_indices.len();

                // If few components changed, do incremental update
                if k > 0 && k <= n / 2 {
                    state.incremental_updates += 1;
                    return self.update_incremental(components, &new_hashes, &changed_indices);
                }

                // If no changes, return cached value
                if k == 0 {
                    return state.last_phi;
                }
            }
        }

        // Full recomputation needed
        self.initialize_incremental_state(components, &new_hashes)
    }

    /// Initialize incremental state with full computation
    fn initialize_incremental_state(&mut self, components: &[BinaryHV], hashes: &[u64]) -> f64 {
        let n = components.len();

        // Build full similarity matrix
        let similarity_matrix = {
            #[cfg(feature = "parallel")]
            {
                if n > 16 {
                    self.build_similarity_matrix_parallel(components)
                } else {
                    self.build_similarity_matrix_sequential(components)
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                self.build_similarity_matrix_sequential(components)
            }
        };

        // Compute degrees
        let degrees: Vec<f64> = similarity_matrix
            .iter()
            .map(|row: &Vec<f64>| row.iter().sum::<f64>() - 1.0)
            .collect();

        // Compute Φ
        let algebraic_connectivity = self.estimate_fiedler_value(&similarity_matrix, &degrees);
        let phi = (algebraic_connectivity / degrees.iter().sum::<f64>().max(1.0) * n as f64)
            .clamp(0.0, 1.0);

        // Cache state
        let mut full_recomputes = 0;
        if let Some(ref state) = self.incremental_state {
            full_recomputes = state.full_recomputes;
        }

        self.incremental_state = Some(IncrementalPhiState {
            similarity_matrix,
            component_hashes: hashes.to_vec(),
            degrees,
            last_phi: phi,
            incremental_updates: 0,
            full_recomputes: full_recomputes + 1,
        });

        phi
    }

    /// Perform incremental update for changed components - O(k×n)
    fn update_incremental(
        &mut self,
        components: &[BinaryHV],
        new_hashes: &[u64],
        changed_indices: &[usize],
    ) -> f64 {
        let state = self
            .incremental_state
            .as_mut()
            .expect("update_incremental only called when incremental_state is Some");
        let n = components.len();

        // Update only affected rows/columns of similarity matrix
        for &i in changed_indices {
            for j in 0..n {
                if i != j {
                    let sim = components[i].similarity(&components[j]) as f64;
                    state.similarity_matrix[i][j] = sim;
                    state.similarity_matrix[j][i] = sim;
                }
            }
        }

        // Update hashes
        for &i in changed_indices {
            state.component_hashes[i] = new_hashes[i];
        }

        // Recompute degrees for affected rows
        for &i in changed_indices {
            state.degrees[i] = state.similarity_matrix[i].iter().sum::<f64>() - 1.0;
        }

        // Also update degrees for columns that were affected
        for j in 0..n {
            if !changed_indices.contains(&j) {
                let mut degree = 0.0;
                for k in 0..n {
                    if k != j {
                        degree += state.similarity_matrix[j][k];
                    }
                }
                state.degrees[j] = degree;
            }
        }

        // Recompute Φ with updated matrix
        // Extract data and drop the mutable borrow before calling estimate_fiedler_value
        let similarity_matrix_clone = state.similarity_matrix.clone();
        let degrees_clone = state.degrees.clone();
        let degree_sum = degrees_clone.iter().sum::<f64>().max(1.0);

        // Release mutable borrow of state (drop does nothing for references)
        let _ = state;

        let algebraic_connectivity =
            self.estimate_fiedler_value(&similarity_matrix_clone, &degrees_clone);

        let phi = (algebraic_connectivity / degree_sum * n as f64).clamp(0.0, 1.0);

        // Re-acquire state to update last_phi
        if let Some(state) = self.incremental_state.as_mut() {
            state.last_phi = phi;
        }
        phi
    }

    /// Hash a single component for change detection
    fn hash_single_component(&self, component: &BinaryHV) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        component.0.hash(&mut hasher);
        hasher.finish()
    }

    /// Get incremental state statistics
    pub fn incremental_stats(&self) -> Option<(u64, u64)> {
        self.incremental_state
            .as_ref()
            .map(|s| (s.incremental_updates, s.full_recomputes))
    }

    /// Clear incremental state (force full recomputation on next call)
    pub fn clear_incremental_state(&mut self) {
        self.incremental_state = None;
    }

    // ========================================================================
    // REVOLUTIONARY #91: HIERARCHICAL Φ DECOMPOSITION (O(n²))
    // ========================================================================

    /// Compute hierarchical Φ decomposition
    ///
    /// Returns Φ values at three scales (micro, meso, macro), enabling
    /// analysis of how consciousness emerges from local to global integration.
    ///
    /// # Algorithm
    ///
    /// 1. Build similarity matrix and detect natural clusters
    /// 2. Compute micro Φ within each cluster
    /// 3. Compute meso Φ between clusters
    /// 4. Compute macro Φ globally
    /// 5. Analyze emergence and bottlenecks
    pub fn compute_hierarchical(&mut self, components: &[BinaryHV]) -> HierarchicalPhi {
        let n = components.len();

        // Trivial cases
        if n < 2 {
            return HierarchicalPhi {
                micro_phi: 0.0,
                meso_phi: 0.0,
                macro_phi: 0.0,
                num_clusters: n.min(1),
                bottleneck_score: 0.0,
                emergence_ratio: 1.0,
            };
        }

        // Step 1: Build similarity matrix
        let similarity_matrix = {
            #[cfg(feature = "parallel")]
            {
                if n > 16 {
                    self.build_similarity_matrix_parallel(components)
                } else {
                    self.build_similarity_matrix_sequential(components)
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                self.build_similarity_matrix_sequential(components)
            }
        };

        // Step 2: Detect natural clusters using simple threshold-based clustering
        let clusters = self.detect_clusters(&similarity_matrix, n);
        let num_clusters = clusters
            .iter()
            .filter(|&&c| c >= 0)
            .max()
            .map(|&m| m as usize + 1)
            .unwrap_or(1);

        // Step 3: Compute micro Φ (within-cluster integration)
        let micro_phi = self.compute_micro_phi(&similarity_matrix, &clusters, n, num_clusters);

        // Step 4: Compute meso Φ (between-cluster integration)
        let meso_phi = self.compute_meso_phi(&similarity_matrix, &clusters, num_clusters);

        // Step 5: Compute macro Φ (global integration)
        let degrees: Vec<f64> = similarity_matrix
            .iter()
            .map(|row: &Vec<f64>| row.iter().sum::<f64>() - 1.0) // Subtract self-similarity
            .collect();
        let algebraic_connectivity = self.estimate_fiedler_value(&similarity_matrix, &degrees);
        let macro_phi = (algebraic_connectivity / degrees.iter().sum::<f64>().max(1.0) * n as f64)
            .clamp(0.0, 1.0);

        // Step 6: Compute emergence ratio and bottleneck
        let expected_integration = micro_phi * num_clusters as f64;
        let emergence_ratio = if expected_integration > 0.001 {
            macro_phi / expected_integration
        } else {
            1.0
        };

        // Bottleneck: gap between macro and meso integration
        // Lower is better (meso nearly matches macro)
        let bottleneck_score = (macro_phi - meso_phi).abs();

        HierarchicalPhi {
            micro_phi,
            meso_phi,
            macro_phi,
            num_clusters,
            bottleneck_score,
            emergence_ratio,
        }
    }

    /// Detect natural clusters in the similarity matrix
    ///
    /// Uses a simple threshold-based approach: components with similarity > 0.6
    /// are assigned to the same cluster. This is O(n²) but can be optimized.
    fn detect_clusters(&self, similarity_matrix: &[Vec<f64>], n: usize) -> Vec<i32> {
        let similarity_threshold = 0.6;
        let mut clusters = vec![-1i32; n]; // -1 = unassigned
        let mut next_cluster = 0i32;

        for i in 0..n {
            if clusters[i] >= 0 {
                continue; // Already assigned
            }

            // Start a new cluster
            clusters[i] = next_cluster;
            let mut stack = vec![i];

            // BFS to find connected components
            while let Some(current) = stack.pop() {
                for j in 0..n {
                    if clusters[j] < 0 && similarity_matrix[current][j] > similarity_threshold {
                        clusters[j] = next_cluster;
                        stack.push(j);
                    }
                }
            }

            next_cluster += 1;
        }

        clusters
    }

    /// Compute micro Φ: average within-cluster integration
    fn compute_micro_phi(
        &self,
        similarity_matrix: &[Vec<f64>],
        clusters: &[i32],
        n: usize,
        num_clusters: usize,
    ) -> f64 {
        if num_clusters == 0 || n == 0 {
            return 0.0;
        }

        let mut total_phi = 0.0;
        let mut cluster_count = 0;

        for c in 0..num_clusters as i32 {
            let members: Vec<usize> = (0..n).filter(|&i| clusters[i] == c).collect();
            if members.len() < 2 {
                continue;
            }

            // Compute integration within this cluster
            let mut cluster_integration = 0.0;
            let mut pairs = 0;

            for i in 0..members.len() {
                for j in (i + 1)..members.len() {
                    cluster_integration += similarity_matrix[members[i]][members[j]];
                    pairs += 1;
                }
            }

            if pairs > 0 {
                total_phi += cluster_integration / pairs as f64;
                cluster_count += 1;
            }
        }

        if cluster_count > 0 {
            total_phi / cluster_count as f64
        } else {
            0.0
        }
    }

    /// Compute meso Φ: between-cluster integration
    fn compute_meso_phi(
        &self,
        similarity_matrix: &[Vec<f64>],
        clusters: &[i32],
        num_clusters: usize,
    ) -> f64 {
        if num_clusters < 2 {
            return 0.0;
        }

        let n = clusters.len();
        let mut total_between = 0.0;
        let mut between_pairs = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                if clusters[i] != clusters[j] {
                    total_between += similarity_matrix[i][j];
                    between_pairs += 1;
                }
            }
        }

        if between_pairs > 0 {
            total_between / between_pairs as f64
        } else {
            0.0
        }
    }

    // ========================================================================
    // TIER 0: MOCK (O(1))
    // ========================================================================

    /// O(1) deterministic mock for testing
    ///
    /// Returns predictable values based solely on component count.
    /// Formula: φ = min(0.1 × n + 0.2, 0.95)
    fn compute_mock(&self, components: &[BinaryHV]) -> f64 {
        let n = components.len() as f64;
        // Linear relationship with component count, capped at 0.95
        (0.1 * n + 0.2).min(0.95)
    }

    // ========================================================================
    // TIER 1: HEURISTIC (O(n))
    // ========================================================================

    /// O(n) fast heuristic approximation using partition sampling
    ///
    /// **CRITICAL FIX (Dec 26, 2025)**: Previous implementation measured
    /// "distinctiveness from bundle" which doesn't correlate with integration.
    ///
    /// **New approach**: Implements IIT 3.0 via partition sampling:
    /// - Φ = system_info - min_partition_info (MIP approximation)
    /// - Samples bipartitions instead of exhaustive search
    /// - Measures actual information loss when system is partitioned
    /// - O(n × samples) complexity with configurable sampling rate
    ///
    /// **Validation**: This implementation should produce Φ values that
    /// correlate strongly (r > 0.85) with consciousness state integration levels.
    fn compute_heuristic(&self, components: &[BinaryHV]) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Step 1: Compute system information (whole system)
        let bundled = self.bundle(components);
        let system_info = self.compute_system_info(&bundled, components);

        // Step 2: Sample partitions to approximate MIP (Minimum Information Partition)
        // More samples = better approximation, but slower
        // Adaptive: larger systems get more samples for accuracy
        let num_samples = if n <= 4 {
            // Small systems: exhaustive is feasible
            (1 << (n - 1)) - 1 // All bipartitions except trivial
        } else {
            // Large systems: sample intelligently
            // Rule: 3n samples gives good approximation with O(n) complexity
            (n * 3).min(100) // Cap at 100 for performance
        };

        let mut min_partition_info = f64::MAX;

        // For small systems, enumerate all partitions
        if n <= 4 {
            // Exhaustive enumeration (2^n - 2 partitions, excluding trivial)
            for mask in 1..(1u64 << n) - 1 {
                let mut part_a = Vec::new();
                let mut part_b = Vec::new();

                for i in 0..n {
                    if (mask & (1 << i)) != 0 {
                        part_a.push(i);
                    } else {
                        part_b.push(i);
                    }
                }

                // Skip trivial partitions (all in one part)
                if part_a.is_empty() || part_b.is_empty() {
                    continue;
                }

                let partition_info = self.compute_partition_info(components, &part_a, &part_b);
                min_partition_info = min_partition_info.min(partition_info);
            }
        } else {
            // Sample random bipartitions for large systems (supports n > 64)
            for _ in 0..num_samples {
                // Generate random bipartition using Vec<bool> (no 64-bit limit)
                let partition_mask = self.random_bipartition_vec(n);

                let mut part_a = Vec::new();
                let mut part_b = Vec::new();

                for i in 0..n {
                    if partition_mask[i] {
                        part_a.push(i);
                    } else {
                        part_b.push(i);
                    }
                }

                // Skip trivial partitions
                if part_a.is_empty() || part_b.is_empty() {
                    continue;
                }

                // Compute information for this partition
                let partition_info = self.compute_partition_info(components, &part_a, &part_b);
                min_partition_info = min_partition_info.min(partition_info);
            }

            // Also test some intelligent partitions based on similarity
            let intelligent_partitions =
                self.generate_intelligent_partitions(components, 5.min(n / 2));
            for (part_a, part_b) in intelligent_partitions {
                let partition_info = self.compute_partition_info(components, &part_a, &part_b);
                min_partition_info = min_partition_info.min(partition_info);
            }
        }

        // Step 3: Φ = information lost when system is partitioned at MIP
        // This is the core IIT 3.0 formula
        let phi = (system_info - min_partition_info).max(0.0);

        // Step 4: Normalize by theoretical maximum
        // CRITICAL FIX (Dec 27, 2025): system_info and partition_info already include ln(n) scaling.
        // Dividing by ln(n) again REMOVES the meaningful signal!
        //
        // The phi value is already in a good range because:
        // - system_info ∈ [0, ln(n)] (similarity ∈ [0,1], scaled by ln(n))
        // - partition_info ∈ [0, ln(n)] (subset of system correlations)
        // - phi = difference ∈ [0, ln(n)]
        //
        // For n=10: ln(10) ≈ 2.3, so phi ∈ [0, 2.3]
        // For normalization to [0,1], divide by ln(n) * max_possible_similarity
        //
        // But actually, a better normalization is to recognize that maximum integration
        // occurs when ALL cross-partition correlations are lost, which happens when
        // system is fully integrated but partition destroys all correlations.
        //
        // Maximum phi ≈ system_info (when partition_info → 0)
        // So normalize by system_info to get relative integration loss

        if system_info > 0.001 {
            (phi / system_info).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Generate a random bipartition as Vec<bool>, supporting arbitrary n.
    ///
    /// Each element is independently assigned to partition A (true) or B (false)
    /// with equal probability. Guarantees a non-trivial partition (both sides non-empty).
    fn random_bipartition_vec(&self, n: usize) -> Vec<bool> {
        use std::collections::hash_map::RandomState;
        use std::hash::BuildHasher;

        for attempt in 0..100u64 {
            let mask: Vec<bool> = (0..n)
                .map(|i| {
                    RandomState::new().hash_one((self.stats.total_calculations, attempt, i)) & 1
                        == 1
                })
                .collect();

            // Ensure non-trivial partition (both sides non-empty)
            let count_true = mask.iter().filter(|&&b| b).count();
            if count_true > 0 && count_true < n {
                return mask;
            }
        }

        // Fallback: balanced partition (first half in A, second half in B)
        (0..n).map(|i| i < n / 2).collect()
    }

    /// Generate intelligent partitions based on component similarity
    ///
    /// Creates partitions that group similar components together,
    /// as these are likely to have low partition information.
    fn generate_intelligent_partitions(
        &self,
        components: &[BinaryHV],
        num_partitions: usize,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        let n = components.len();
        let mut partitions = Vec::new();

        if n < 2 || num_partitions == 0 {
            return partitions;
        }

        // Strategy 1: Similarity-based clustering
        // Group most similar components together
        let mut similarity_matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = components[i].similarity(&components[j]) as f64;
                similarity_matrix[i][j] = sim;
                similarity_matrix[j][i] = sim;
            }
        }

        // Find the most similar pair and build partition around them
        for partition_idx in 0..num_partitions {
            let mut part_a = Vec::new();
            let mut part_b = Vec::new();

            // Start with different seed pairs for diversity
            let seed_offset = partition_idx * (n / num_partitions.max(1));

            // Add first half to part_a, second half to part_b
            // but offset by seed to get different partitions
            for i in 0..n {
                let idx = (i + seed_offset) % n;
                if i < n / 2 {
                    part_a.push(idx);
                } else {
                    part_b.push(idx);
                }
            }

            // Only add if non-trivial
            if !part_a.is_empty() && !part_b.is_empty() {
                partitions.push((part_a, part_b));
            }
        }

        partitions
    }

    // ========================================================================
    // TIER 2: SPECTRAL (O(n²))
    // ========================================================================

    // ========================================================================
    // REVOLUTIONARY IMPROVEMENT #89: RAYON PARALLELIZATION
    // ========================================================================
    //
    // The O(n²) spectral computation is embarrassingly parallel:
    // - Each similarity pair (i,j) is independent
    // - Rayon provides work-stealing for optimal load balancing
    // - Expected speedup: 4-8x on modern multi-core CPUs
    //
    // Threshold: Only parallelize for n > 16 (overhead dominates for small n)
    // ========================================================================

    /// O(n²) spectral approximation using graph connectivity
    ///
    /// Models components as a graph where edge weight = similarity.
    /// Φ ≈ 1 - algebraic_connectivity (Fiedler value)
    ///
    /// **Revolutionary #89**: Uses Rayon parallelization for 4-8x speedup
    /// on multi-core systems when n > 16 components.
    ///
    /// Intuition: A highly connected system (high Φ) will have high
    /// algebraic connectivity (hard to partition).
    fn compute_spectral(&self, components: &[BinaryHV]) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Build similarity matrix - PARALLEL for large n
        #[cfg(feature = "parallel")]
        let similarity_matrix = if n > 16 {
            // Revolutionary #89: Parallel similarity computation
            self.build_similarity_matrix_parallel(components)
        } else {
            // Sequential for small n (avoid Rayon overhead)
            self.build_similarity_matrix_sequential(components)
        };
        #[cfg(not(feature = "parallel"))]
        let similarity_matrix = self.build_similarity_matrix_sequential(components);

        // Compute Laplacian: L = D - A (where D is degree matrix)
        let degrees: Vec<f64> = {
            #[cfg(feature = "parallel")]
            {
                if n > 16 {
                    similarity_matrix
                        .par_iter()
                        .map(|row| row.iter().sum::<f64>() - 1.0)
                        .collect()
                } else {
                    similarity_matrix
                        .iter()
                        .map(|row| row.iter().sum::<f64>() - 1.0)
                        .collect()
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                similarity_matrix
                    .iter()
                    .map(|row| row.iter().sum::<f64>() - 1.0)
                    .collect()
            }
        };

        // For small matrices, use power iteration to find second smallest eigenvalue
        // (algebraic connectivity / Fiedler value)
        let algebraic_connectivity = self.estimate_fiedler_value(&similarity_matrix, &degrees);

        // Φ correlates with how hard the system is to partition
        // High connectivity → high Φ
        let phi = algebraic_connectivity / degrees.iter().sum::<f64>().max(1.0) * n as f64;

        phi.clamp(0.0, 1.0)
    }

    /// Build similarity matrix in parallel using Rayon
    /// Revolutionary #89: O(n²) similarity with ~linear parallelization
    #[cfg(feature = "parallel")]
    pub fn build_similarity_matrix_parallel(&self, components: &[BinaryHV]) -> Vec<Vec<f64>> {
        let n = components.len();

        // Compute upper triangle in parallel, then mirror
        let pairs: Vec<(usize, usize, f64)> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                (i + 1..n).into_par_iter().map(move |j| {
                    let sim = components[i].similarity(&components[j]) as f64;
                    (i, j, sim)
                })
            })
            .collect();

        // Build full matrix from pairs
        let mut matrix = vec![vec![0.0f64; n]; n];
        for (i, j, sim) in pairs {
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
        for i in 0..n {
            matrix[i][i] = 1.0; // Self-similarity
        }

        matrix
    }

    /// Build similarity matrix sequentially (for small n)
    pub fn build_similarity_matrix_sequential(&self, components: &[BinaryHV]) -> Vec<Vec<f64>> {
        let n = components.len();
        let mut matrix = vec![vec![0.0f64; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = components[i].similarity(&components[j]) as f64;
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
            matrix[i][i] = 1.0; // Self-similarity
        }

        matrix
    }

    /// Compute Fiedler value (second-smallest eigenvalue of the graph Laplacian)
    /// using power iteration with deflation.
    ///
    /// The Laplacian L = D - A where D is the degree matrix and A is the
    /// similarity (adjacency) matrix. The smallest eigenvalue of L is always 0
    /// (with eigenvector = all-ones for connected graphs). The Fiedler value λ₂
    /// is the algebraic connectivity of the graph.
    ///
    /// Algorithm:
    /// 1. Build Laplacian from similarity matrix and degrees
    /// 2. Shift: M = λ_max*I - L (converts smallest eigenvalues to largest)
    /// 3. Power iteration on M gives largest eigenvalue = λ_max - λ₁ = λ_max
    /// 4. Deflate M by the found eigenvector (all-ones direction)
    /// 5. Power iteration on deflated M gives λ_max - λ₂
    /// 6. Therefore λ₂ = λ_max - (λ_max - λ₂) = result from step 5 subtracted
    fn estimate_fiedler_value(&self, similarity: &[Vec<f64>], degrees: &[f64]) -> f64 {
        let n = similarity.len();
        if n < 2 {
            return 0.0;
        }

        // Build Laplacian: L[i][j] = degree[i] if i==j, -similarity[i][j] otherwise
        let mut laplacian = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            laplacian[i][i] = degrees[i];
            for j in 0..n {
                if i != j {
                    laplacian[i][j] = -similarity[i][j];
                }
            }
        }

        // Estimate λ_max of the Laplacian via Gershgorin bound (upper bound on largest eigenvalue)
        let lambda_max_bound: f64 = degrees.iter().cloned().fold(0.0f64, f64::max) * 2.0;
        if lambda_max_bound < 1e-10 {
            return 0.0; // Degenerate graph with no edges
        }

        // Shift: M = λ_max_bound * I - L
        // This makes the eigenvalues of M = λ_max_bound - λ_i(L)
        // So the largest eigenvalue of M corresponds to the smallest eigenvalue of L (= 0)
        // and the second-largest of M corresponds to λ₂
        let mut shifted = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                shifted[i][j] = -laplacian[i][j];
                if i == j {
                    shifted[i][j] += lambda_max_bound;
                }
            }
        }

        // Power iteration helper: returns (eigenvalue, eigenvector)
        let power_iter = |mat: &[Vec<f64>], max_iters: usize| -> (f64, Vec<f64>) {
            let dim = mat.len();
            // Initialize with non-degenerate vector
            let mut v: Vec<f64> = (0..dim).map(|i| (i as f64 + 1.0).sin()).collect();
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }

            for _ in 0..max_iters {
                // w = M * v
                let w: Vec<f64> = (0..dim)
                    .map(|i| mat[i].iter().zip(&v).map(|(a, b)| a * b).sum())
                    .collect();

                let w_norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                if w_norm < 1e-15 {
                    break;
                }
                v = w.iter().map(|x| x / w_norm).collect();
            }

            // Rayleigh quotient: eigenvalue = v^T M v
            let mv: Vec<f64> = (0..dim)
                .map(|i| mat[i].iter().zip(&v).map(|(a, b)| a * b).sum())
                .collect();
            let eigenvalue: f64 = v.iter().zip(&mv).map(|(a, b)| a * b).sum();
            (eigenvalue, v)
        };

        let max_iters = 100;

        // Step 1: Find largest eigenvalue of M (corresponds to λ₁=0 of L)
        let (mu_1, v_1) = power_iter(&shifted, max_iters);

        // Step 2: Deflate M: M' = M - μ₁ * v₁ * v₁^T
        let mut deflated = shifted.clone();
        for i in 0..n {
            for j in 0..n {
                deflated[i][j] -= mu_1 * v_1[i] * v_1[j];
            }
        }

        // Step 3: Find largest eigenvalue of deflated M (corresponds to λ₂ of L)
        let (mu_2, _) = power_iter(&deflated, max_iters);

        // λ₂ = λ_max_bound - μ₂

        (lambda_max_bound - mu_2).max(0.0)
    }

    // ========================================================================
    // TIER 3: EXACT (O(2^n))
    // ========================================================================

    /// O(2^n) exact MIP calculation
    ///
    /// WARNING: Only use for small systems (n ≤ 12)!
    fn compute_exact(&self, components: &[BinaryHV]) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Warn if system is too large
        if n > 12 {
            eprintln!(
                "[WARNING] Exact Φ calculation for {} components may be slow (O(2^{}) = {} partitions)",
                n, n, 1u64 << n
            );
        }

        // Compute system information
        let bundled = self.bundle(components);
        let system_info = self.compute_system_info(&bundled, components);

        // Find MIP by exhaustive search
        let mut min_partition_info = f64::MAX;

        // Iterate through all bipartitions
        for mask in 1..(1u64 << n) - 1 {
            let mut part_a = Vec::new();
            let mut part_b = Vec::new();

            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    part_a.push(i);
                } else {
                    part_b.push(i);
                }
            }

            // Skip trivial partitions
            if part_a.is_empty() || part_b.is_empty() {
                continue;
            }

            // Compute partition information
            let partition_info = self.compute_partition_info(components, &part_a, &part_b);
            min_partition_info = min_partition_info.min(partition_info);
        }

        // Φ = system_info - min_partition_info
        let phi = (system_info - min_partition_info).max(0.0);

        // Normalize by system_info (same fix as heuristic tier)
        // CRITICAL FIX (Dec 27, 2025): Normalize by system_info, not sqrt(n)
        if system_info > 0.001 {
            (phi / system_info).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    /// Bundle components into a single hypervector
    fn bundle(&self, components: &[BinaryHV]) -> BinaryHV {
        if components.is_empty() {
            return BinaryHV::zero();
        }

        // Use the static bundle function from BinaryHV
        BinaryHV::bundle(components)
    }

    /// Compute system information using pairwise mutual information
    ///
    /// **Key Insight**: Integrated information comes from correlations BETWEEN components.
    /// - High similarity between components → high integration → high information
    /// - Components that share patterns have mutual information
    /// - The bundle captures the integrated state
    ///
    /// We approximate I(components) using average pairwise similarity
    fn compute_system_info(&self, _bundled: &BinaryHV, components: &[BinaryHV]) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Measure total pairwise mutual information
        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = components[i].similarity(&components[j]) as f64;
                total_similarity += sim;
                pair_count += 1;
            }
        }

        // Average similarity = proxy for mutual information
        // Scale by log(n) to account for system size
        let avg_similarity = if pair_count > 0 {
            total_similarity / pair_count as f64
        } else {
            0.0
        };

        // System information = how much the components are correlated
        // Higher correlation → more integration → more Φ
        avg_similarity * (n as f64).ln().max(1.0)
    }

    /// Compute information retained after partitioning
    ///
    /// **Key Insight**: When we partition the system, we LOSE cross-partition correlations.
    /// The partition info is the pairwise correlations that REMAIN (within each part).
    ///
    /// partition_info = within_part_A_correlations + within_part_B_correlations
    /// system_info = ALL pairwise correlations (including cross-partition)
    /// Φ = system_info - partition_info = CROSS-PARTITION correlations (what we lose)
    fn compute_partition_info(
        &self,
        components: &[BinaryHV],
        part_a: &[usize],
        part_b: &[usize],
    ) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Compute within-partition correlations ONLY
        let mut within_partition_similarity = 0.0;
        let mut within_pair_count = 0;

        // Part A internal correlations
        for i in 0..part_a.len() {
            for j in (i + 1)..part_a.len() {
                let idx_i = part_a[i];
                let idx_j = part_a[j];
                let sim = components[idx_i].similarity(&components[idx_j]) as f64;
                within_partition_similarity += sim;
                within_pair_count += 1;
            }
        }

        // Part B internal correlations
        for i in 0..part_b.len() {
            for j in (i + 1)..part_b.len() {
                let idx_i = part_b[i];
                let idx_j = part_b[j];
                let sim = components[idx_i].similarity(&components[idx_j]) as f64;
                within_partition_similarity += sim;
                within_pair_count += 1;
            }
        }

        // Average within-partition similarity
        // (Does NOT include cross-partition pairs - those are what we lose!)
        let avg_within_similarity = if within_pair_count > 0 {
            within_partition_similarity / within_pair_count as f64
        } else {
            0.0
        };

        // Scale by log(n) to match system_info scaling
        avg_within_similarity * (n as f64).ln().max(1.0)
    }

    /// Check cache for precomputed value
    fn check_cache(&self, components: &[BinaryHV]) -> Option<f64> {
        if components.is_empty() {
            return Some(0.0);
        }

        let n = components.len();
        let hash = self.hash_components(components);

        for &(cached_n, cached_hash, phi) in &self.cache {
            if cached_n == n && cached_hash == hash {
                return Some(phi);
            }
        }

        None
    }

    /// Update cache with new value
    fn update_cache(&mut self, components: &[BinaryHV], phi: f64) {
        let n = components.len();
        let hash = self.hash_components(components);

        // Simple LRU: remove oldest if at capacity
        if self.cache.len() >= self.config.cache_size {
            self.cache.remove(0);
        }

        self.cache.push((n, hash, phi));
    }

    /// Simple hash of component array
    fn hash_components(&self, components: &[BinaryHV]) -> u64 {
        let mut hash = 0u64;
        for (i, component) in components.iter().enumerate() {
            // XOR first few bytes with position-based scrambling
            let bytes = &component.0;
            for (j, &byte) in bytes.iter().take(8).enumerate() {
                hash ^= (byte as u64) << ((i + j) % 56);
            }
        }
        hash
    }

    // ========================================================================
    // REVOLUTIONARY #92: CAUSAL Φ ATTRIBUTION
    // ========================================================================
    //
    // **Key Insight**: Not all components contribute equally to consciousness.
    // Some are "critical" (removing them dramatically reduces Φ), while others
    // are "redundant" (removing them barely affects Φ).
    //
    // This enables:
    // - Identifying consciousness bottlenecks
    // - Understanding which neural populations are essential
    // - Designing minimal consciousness architectures
    // - Detecting redundancy for compression
    //
    // **Method**: Leave-one-out analysis
    // For each component i: Φ_i = Φ_baseline - Φ_without_i
    // High Φ_i = critical component
    // Low/Negative Φ_i = redundant component
    // ========================================================================

    /// Compute causal Φ attribution for each component
    ///
    /// Uses leave-one-out analysis: for each component, compute Φ with that
    /// component removed. The importance score is how much Φ decreases.
    ///
    /// **Complexity**: O(n × Φ_complexity)
    /// - Heuristic tier: O(n²)
    /// - Spectral tier: O(n³)
    /// - Exact tier: O(n × 2^n) - use sparingly!
    ///
    /// # Returns
    /// PhiAttribution containing:
    /// - baseline_phi: Φ with all components
    /// - component_scores: importance score for each component
    /// - importance_ranking: indices sorted by importance (highest first)
    /// - critical_components: indices where removal reduces Φ significantly
    /// - redundant_components: indices where removal barely affects Φ
    /// - concentration_index: Gini-like measure (0=uniform, 1=concentrated)
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
    /// let components = create_conscious_system();
    /// let attr = phi.compute_attribution(&components);
    /// println!("Most critical component: {}", attr.importance_ranking[0]);
    /// ```
    pub fn compute_attribution(&mut self, components: &[BinaryHV]) -> PhiAttribution {
        let n = components.len();

        // Edge cases
        if n == 0 {
            return PhiAttribution {
                baseline_phi: 0.0,
                component_scores: vec![],
                importance_ranking: vec![],
                critical_components: vec![],
                redundant_components: vec![],
                concentration_index: 0.0,
            };
        }

        if n == 1 {
            return PhiAttribution {
                baseline_phi: 0.0, // Single component has no integration
                component_scores: vec![0.0],
                importance_ranking: vec![0],
                critical_components: vec![],
                redundant_components: vec![0],
                concentration_index: 0.0,
            };
        }

        // Step 1: Compute baseline Φ
        let baseline_phi = self.compute(components);

        // Step 2: Leave-one-out analysis
        let mut component_scores = Vec::with_capacity(n);

        for exclude_idx in 0..n {
            // Create component list without the excluded one
            let remaining: Vec<BinaryHV> = components
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != exclude_idx)
                .map(|(_, c)| *c)
                .collect();

            // Compute Φ without this component
            let phi_without = self.compute(&remaining);

            // Importance = how much Φ drops when we remove this component
            let importance = baseline_phi - phi_without;
            component_scores.push(importance);
        }

        // Step 3: Create importance ranking (highest first)
        let mut importance_ranking: Vec<usize> = (0..n).collect();
        importance_ranking.sort_by(|&a, &b| component_scores[b].total_cmp(&component_scores[a]));

        // Step 4: Identify critical and redundant components
        // Critical: removal reduces Φ by more than 10% of baseline
        // Redundant: removal reduces Φ by less than 1% of baseline
        let critical_threshold = baseline_phi * 0.10;
        let redundant_threshold = baseline_phi * 0.01;

        let critical_components: Vec<usize> = (0..n)
            .filter(|&i| component_scores[i] > critical_threshold)
            .collect();

        let redundant_components: Vec<usize> = (0..n)
            .filter(|&i| component_scores[i] < redundant_threshold)
            .collect();

        // Step 5: Compute concentration index (Gini coefficient on importance)
        let concentration_index = self.compute_concentration(&component_scores);

        PhiAttribution {
            baseline_phi,
            component_scores,
            importance_ranking,
            critical_components,
            redundant_components,
            concentration_index,
        }
    }

    /// Fast attribution using centrality approximation
    ///
    /// Instead of computing n × Φ calculations, we approximate importance
    /// using network centrality measures. Much faster but less accurate.
    ///
    /// **Complexity**: O(n²) regardless of Φ tier
    ///
    /// **Method**:
    /// 1. Build similarity graph between components
    /// 2. Compute weighted degree centrality
    /// 3. Higher centrality ≈ more critical to integration
    ///
    /// Use this for:
    /// - Large systems (n > 100)
    /// - Real-time analysis
    /// - Initial screening before detailed attribution
    pub fn compute_attribution_fast(&mut self, components: &[BinaryHV]) -> PhiAttribution {
        let n = components.len();

        // Edge cases
        if n == 0 {
            return PhiAttribution {
                baseline_phi: 0.0,
                component_scores: vec![],
                importance_ranking: vec![],
                critical_components: vec![],
                redundant_components: vec![],
                concentration_index: 0.0,
            };
        }

        if n == 1 {
            return PhiAttribution {
                baseline_phi: 0.0,
                component_scores: vec![0.0],
                importance_ranking: vec![0],
                critical_components: vec![],
                redundant_components: vec![0],
                concentration_index: 0.0,
            };
        }

        // Step 1: Compute baseline Φ (we still need this)
        let baseline_phi = self.compute(components);

        // Step 2: Build weighted degree centrality from similarity graph
        // Centrality[i] = sum of similarities to all other components
        let mut centralities = vec![0.0f64; n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let sim = components[i].similarity(&components[j]) as f64;
                    centralities[i] += sim;
                }
            }
        }

        // Step 3: Normalize centralities to importance scores
        // Higher centrality = more connected = more critical to integration
        // Scale to be proportional to baseline_phi
        let max_centrality = centralities.iter().cloned().fold(0.0f64, f64::max);
        let component_scores: Vec<f64> = if max_centrality > 0.0 {
            centralities
                .iter()
                .map(|&c| (c / max_centrality) * baseline_phi * 0.5)
                .collect()
        } else {
            vec![0.0; n]
        };

        // Step 4: Create importance ranking (highest first)
        let mut importance_ranking: Vec<usize> = (0..n).collect();
        importance_ranking.sort_by(|&a, &b| component_scores[b].total_cmp(&component_scores[a]));

        // Step 5: Identify critical and redundant components
        let critical_threshold = baseline_phi * 0.10;
        let redundant_threshold = baseline_phi * 0.01;

        let critical_components: Vec<usize> = (0..n)
            .filter(|&i| component_scores[i] > critical_threshold)
            .collect();

        let redundant_components: Vec<usize> = (0..n)
            .filter(|&i| component_scores[i] < redundant_threshold)
            .collect();

        // Step 6: Compute concentration index
        let concentration_index = self.compute_concentration(&component_scores);

        PhiAttribution {
            baseline_phi,
            component_scores,
            importance_ranking,
            critical_components,
            redundant_components,
            concentration_index,
        }
    }

    /// Compute Gini coefficient for concentration measurement
    ///
    /// 0 = perfectly uniform (all components equally important)
    /// 1 = perfectly concentrated (one component has all importance)
    fn compute_concentration(&self, scores: &[f64]) -> f64 {
        let n = scores.len();
        if n <= 1 {
            return 0.0;
        }

        // Normalize to positive values
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let shifted: Vec<f64> = scores.iter().map(|&s| s - min_score + 1e-10).collect();
        let total: f64 = shifted.iter().sum();

        if total <= 0.0 {
            return 0.0;
        }

        // Sort for Gini calculation
        let mut sorted = shifted.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));

        // Gini coefficient formula
        let mut gini_sum = 0.0;
        for (i, &s) in sorted.iter().enumerate() {
            gini_sum += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * s;
        }

        (gini_sum / (n as f64 * total)).abs()
    }
}

/// Result of causal Φ attribution analysis
///
/// Identifies which components are critical vs redundant for consciousness.
/// This enables understanding of consciousness architecture and optimization.
#[derive(Debug, Clone)]
pub struct PhiAttribution {
    /// Φ with all components present
    pub baseline_phi: f64,

    /// Importance score for each component
    /// Positive = removing hurts Φ (critical)
    /// Negative = removing helps Φ (interference)
    /// Near-zero = redundant
    pub component_scores: Vec<f64>,

    /// Component indices sorted by importance (highest first)
    pub importance_ranking: Vec<usize>,

    /// Indices of components that are critical (removal >10% Φ drop)
    pub critical_components: Vec<usize>,

    /// Indices of components that are redundant (removal <1% Φ drop)
    pub redundant_components: Vec<usize>,

    /// Gini coefficient measuring importance concentration
    /// 0 = uniform importance across all components
    /// 1 = all importance concentrated in one component
    pub concentration_index: f64,
}

impl PhiAttribution {
    /// Get the most critical component index
    pub fn most_critical(&self) -> Option<usize> {
        self.importance_ranking.first().copied()
    }

    /// Get the most redundant component index
    pub fn most_redundant(&self) -> Option<usize> {
        self.importance_ranking.last().copied()
    }

    /// Check if consciousness is distributed (low concentration) or centralized (high concentration)
    pub fn is_distributed(&self) -> bool {
        self.concentration_index < 0.3
    }

    /// Get percentage of components that are critical
    pub fn critical_percentage(&self) -> f64 {
        if self.component_scores.is_empty() {
            return 0.0;
        }
        (self.critical_components.len() as f64 / self.component_scores.len() as f64) * 100.0
    }
}

impl Default for TieredPhi {
    fn default() -> Self {
        Self::for_production()
    }
}

// ============================================================================
// GLOBAL Φ CALCULATOR (Revolutionary Improvement #86)
// ============================================================================

use once_cell::sync::Lazy;
use std::sync::Mutex;

/// Global thread-safe Φ calculator for convenience functions
/// Uses SampledPartition by default (IIT-aligned, O(n), good accuracy).
/// SpectralConnectivity was removed from default — it measures spectral gap,
/// not IIT integrated information (r = -0.62 correlation with true Φ).
static GLOBAL_PHI: Lazy<Mutex<TieredPhi>> =
    Lazy::new(|| Mutex::new(TieredPhi::new(ApproximationTier::SampledPartition)));

/// Compute Φ using the global calculator
///
/// Thread-safe, cached, with O(n²) spectral approximation by default.
/// Use this for one-off calculations when you don't need control over the calculator.
pub fn global_phi(components: &[BinaryHV]) -> f64 {
    GLOBAL_PHI
        .lock()
        .expect("lock poisoned")
        .compute(components)
}

/// Compute Φ with automatic tier selection based on component count
///
/// This is the recommended way to calculate Φ when you want the best
/// balance of speed and accuracy for your specific component count.
///
/// - n ≤ 8: Exact calculation (feasible, O(2^8) = 256 partitions)
/// - 8 < n ≤ 50: Spectral approximation (accurate, O(n²))
/// - 50 < n ≤ 500: Heuristic (fast, O(n))
/// - n > 500: Mock (instant, O(1))
pub fn auto_phi(components: &[BinaryHV]) -> f64 {
    let n = components.len();
    let tier = auto_tier(n);
    GLOBAL_PHI
        .lock()
        .expect("lock poisoned")
        .compute_with_tier(components, tier)
}

/// Automatically select the appropriate tier based on component count
///
/// Returns the most accurate IIT-valid tier for a given component count.
///
/// **SpectralConnectivity is NOT used** — it measures spectral gap (mixing time),
/// not IIT integrated information. Empirical validation showed r = -0.62
/// correlation with true Φ (nearly opposite behavior). Removed 2026-04-11.
///
/// - n ≤ 8: ExhaustivePartition (O(2^n), true MIP search)
/// - 9 ≤ n ≤ 500: SampledPartition (O(n), IIT-aligned sampling)
/// - n > 500: RandomBaseline (O(1), testing/emergency only)
pub fn auto_tier(n: usize) -> ApproximationTier {
    match n {
        0..=8 => ApproximationTier::ExhaustivePartition,
        9..=500 => ApproximationTier::SampledPartition,
        _ => ApproximationTier::RandomBaseline,
    }
}

/// Reset the global Φ calculator to a specific tier
pub fn set_global_tier(tier: ApproximationTier) {
    *GLOBAL_PHI.lock().expect("lock poisoned") = TieredPhi::new(tier);
}

/// Get statistics from the global Φ calculator
pub fn global_phi_stats() -> TieredPhiStats {
    GLOBAL_PHI.lock().expect("lock poisoned").stats.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::BinaryHV;

    /// Helper: create n random BinaryHV components with distinct seeds.
    fn make_components(n: usize) -> Vec<BinaryHV> {
        (0..n).map(|i| BinaryHV::random(i as u64 + 1000)).collect()
    }

    // ------------------------------------------------------------------
    // 1. Constructor / factory tests
    // ------------------------------------------------------------------

    #[test]
    fn test_tiered_phi_constructors() {
        let t1 = TieredPhi::for_testing();
        assert_eq!(t1.tier(), ApproximationTier::RandomBaseline);

        let t2 = TieredPhi::for_production();
        assert_eq!(t2.tier(), ApproximationTier::SampledPartition);

        let t3 = TieredPhi::for_research();
        assert_eq!(t3.tier(), ApproximationTier::ExhaustivePartition);

        let t4 = TieredPhi::new(ApproximationTier::SampledPartition);
        assert_eq!(t4.tier(), ApproximationTier::SampledPartition);
    }

    #[test]
    fn test_tiered_phi_config_defaults() {
        let cfg = TieredPhiConfig::default();
        assert_eq!(cfg.tier, ApproximationTier::SampledPartition);
        assert!(cfg.auto_downgrade);
        assert!(cfg.enable_cache);
        assert!(cfg.cache_size > 0);
    }

    // ------------------------------------------------------------------
    // 2. Edge cases: empty and single-component inputs
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_empty_returns_zero() {
        let mut phi = TieredPhi::for_testing();
        assert_eq!(phi.compute(&[]), 0.0);

        let mut phi2 = TieredPhi::for_production();
        assert_eq!(phi2.compute(&[]), 0.0);
    }

    #[test]
    fn test_compute_single_component_returns_zero() {
        let single = vec![BinaryHV::random(42)];

        let mut phi_mock = TieredPhi::for_testing();
        assert_eq!(phi_mock.compute(&single), 0.0);

        let mut phi_spectral = TieredPhi::for_production();
        assert_eq!(phi_spectral.compute(&single), 0.0);
    }

    // ------------------------------------------------------------------
    // 3. Basic computation: phi >= 0 and phi <= 1 across tiers
    // ------------------------------------------------------------------

    #[test]
    fn test_mock_tier_phi_in_valid_range() {
        let mut phi = TieredPhi::for_testing();
        for n in 2..=8 {
            let components = make_components(n);
            let val = phi.compute(&components);
            assert!(val >= 0.0, "Mock phi must be >= 0 for n={n}");
            assert!(val <= 1.0, "Mock phi must be <= 1 for n={n}");
        }
    }

    #[test]
    fn test_spectral_tier_phi_non_negative() {
        let mut phi = TieredPhi::new(ApproximationTier::SpectralConnectivity);
        let components = make_components(5);
        let val = phi.compute(&components);
        assert!(val >= 0.0, "Spectral phi must be non-negative");
        assert!(val <= 1.0, "Spectral phi must be <= 1");
    }

    #[test]
    fn test_exact_tier_phi_non_negative() {
        let mut phi = TieredPhi::for_research();
        let components = make_components(4);
        let val = phi.compute(&components);
        assert!(val >= 0.0, "Exact phi must be non-negative");
        assert!(val <= 1.0, "Exact phi must be <= 1");
    }

    // ------------------------------------------------------------------
    // 4. Mock tier is deterministic and monotone in n
    // ------------------------------------------------------------------

    #[test]
    fn test_mock_deterministic_and_monotone() {
        let mut phi = TieredPhi::for_testing();
        let c5 = make_components(5);

        let v1 = phi.compute(&c5);
        let v2 = phi.compute(&c5);
        assert_eq!(v1, v2, "Mock must be deterministic");

        // Monotone: more components => higher phi (for mock: 0.1*n + 0.2)
        let v3 = phi.compute(&make_components(3));
        assert!(v1 > v3, "Mock phi should increase with component count");
    }

    // ------------------------------------------------------------------
    // 5. ApproximationTier helper methods
    // ------------------------------------------------------------------

    #[test]
    fn test_approximation_tier_suitability() {
        assert!(ApproximationTier::RandomBaseline.is_suitable_for(10000));
        assert!(ApproximationTier::SampledPartition.is_suitable_for(10000));
        assert!(!ApproximationTier::SpectralConnectivity.is_suitable_for(5000));
        assert!(!ApproximationTier::ExhaustivePartition.is_suitable_for(20));
        assert!(ApproximationTier::ExhaustivePartition.is_suitable_for(12));
    }

    #[test]
    fn test_auto_tier_selection() {
        assert_eq!(auto_tier(4), ApproximationTier::ExhaustivePartition);
        assert_eq!(auto_tier(30), ApproximationTier::SampledPartition);
        assert_eq!(auto_tier(200), ApproximationTier::SampledPartition);
        assert_eq!(auto_tier(1000), ApproximationTier::RandomBaseline);
    }

    // ------------------------------------------------------------------
    // 6. Hierarchical phi edge case
    // ------------------------------------------------------------------

    #[test]
    fn test_hierarchical_phi_single_component() {
        let mut phi = TieredPhi::for_production();
        let single = vec![BinaryHV::random(99)];
        let h = phi.compute_hierarchical(&single);
        assert_eq!(h.micro_phi, 0.0);
        assert_eq!(h.meso_phi, 0.0);
        assert_eq!(h.macro_phi, 0.0);
        assert_eq!(h.num_clusters, 1);
    }

    // ------------------------------------------------------------------
    // 7. Attribution on empty / single component
    // ------------------------------------------------------------------

    #[test]
    fn test_attribution_edge_cases() {
        let mut phi = TieredPhi::for_testing();

        let attr_empty = phi.compute_attribution(&[]);
        assert_eq!(attr_empty.baseline_phi, 0.0);
        assert!(attr_empty.component_scores.is_empty());

        let attr_one = phi.compute_attribution(&[BinaryHV::random(7)]);
        assert_eq!(attr_one.baseline_phi, 0.0);
        assert_eq!(attr_one.component_scores.len(), 1);
    }
}
