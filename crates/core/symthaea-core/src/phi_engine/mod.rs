// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Φ (Phi) Engine - Integrated Information Measurement Framework
//!
//! ## Purpose
//! PhiEngine is a dedicated framework for measuring integrated information (Φ)
//! in network topologies, providing multiple calculation methods and comprehensive
//! validation tools for consciousness research.
//!
//! ## Theoretical Basis
//! Based on Integrated Information Theory (IIT) by Giulio Tononi, which posits that
//! consciousness corresponds to integrated information - information that is both
//! differentiated (components are distinct) and integrated (cannot be reduced to parts).
//!
//! ## Key Types
//!
//! - [`PhiCalculator`] - Unified trait for all Φ calculation methods
//! - [`PhiResult`] - Standard result structure with Φ value and metadata
//! - [`PhiUncertainty`] - Statistical uncertainty for Φ measurements
//! - [`Complexity`] - Computational complexity classification
//!
//! ## Calculation Methods
//!
//! | Method | Complexity | Accuracy | Use Case |
//! |--------|-----------|----------|----------|
//! | Continuous (`ContinuousPhiCalculator` = `ConnectivityCalculator`, algebraic connectivity λ₂) | O(n³) | **Deprecated: r = -0.62 with true Φ** (`hdc/tiered_phi/core.rs`) | Do not use for Φ; topology-connectivity heuristic only |
//! | Resonator (`ResonantPhiCalculator`) | O(n log n) | **Not a valid Φ approximation** — its own module doc (`hdc/phi_resonant.rs`) says it measures coupled-oscillator resonance dynamics, not IIT integration, and internally uses the same deprecated spectral gap | Modeling synchronization dynamics only |
//! | Tiered (`TieredPhi`) | O(1) to O(2ⁿ) | Variable — `SampledPartition`: r=0.9998 vs exact for small N; `ExhaustivePartition`: true IIT Φ for n≤12; `RandomBaseline`: testing mock, not a measurement; its own `SpectralConnectivity` tier carries the same r=-0.62 caveat as Continuous above | Prefer `SampledPartition`/`ExhaustivePartition` for anything that needs to approximate real Φ |
//!
//! (Corrected 2026-07-03 — this table previously rated the two deprecated/invalid
//! methods "High"/"Medium" accuracy with no caveat.)
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea::phi_engine::{PhiCalculator, ContinuousPhiCalculator};
//! use symthaea::hdc::consciousness_topology_generators as topologies;
//!
//! // Create a topology
//! let ring = topologies::ring_topology(8, 16384, 42);
//!
//! // Calculate Φ using continuous method
//! let calculator = ContinuousPhiCalculator::new();
//! let result = calculator.compute_from_hvs(&ring.node_representations);
//!
//! println!("Φ = {:.4} ({})", result.phi, result.method);
//! ```
//!
//! ## Major Discoveries
//!
//! - **Asymptotic Limit**: Φ → 0.5 as hypercube dimension → ∞
//! - **4D Hypercube Champion**: Highest Φ among 19 tested topologies
//! - **Dimensional Invariance**: Uniform k-regular structures maintain Φ across dimensions
//!
//! ## Related Modules
//!
//! - [`crate::hdc::consciousness_topology_generators`] - 19 topology generators
//! - [`crate::consciousness::consciousness_equation_v2`] - Master Consciousness Equation
//! - [`crate::hdc::unified_hv`] - Unified hypervector types

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::phi_resonant::ResonantPhiCalculator;
use crate::hdc::phi_topology_validation::real_hv_to_hv16;
use crate::hdc::unified_hv::ContinuousHV;

mod cache;
mod calculator;
mod result;

// Re-export main types
pub use cache::{CacheStats, CachedPhiEngine};
pub use calculator::{Complexity, PhiCalculator};
pub use result::{PhiCategory, PhiResult, PhiUncertainty};

// Re-export specific implementations (from existing hdc module)
pub use crate::hdc::spectral_connectivity::ConnectivityCalculator as ContinuousPhiCalculator;
pub use crate::hdc::tiered_phi::{ApproximationTier, TieredPhi, TieredPhiConfig};
// Re-export resonator when available:
// pub use crate::hdc::phi_resonant::ResonatorPhi;

/// PhiEngine facade for easy method selection and comparison
#[derive(Clone)]
pub struct PhiEngine {
    /// Currently selected method
    method: PhiMethod,
}

/// Available Φ calculation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhiMethod {
    /// Spectral connectivity (ContinuousHV-based, algebraic connectivity / λ₂).
    ///
    /// This computes the second-smallest eigenvalue of the Laplacian of the
    /// cosine-similarity graph — a measure of how connected the network is.
    /// Previously named "Continuous"; renamed for scientific accuracy since
    /// this is NOT IIT Φ but rather the Fiedler value (λ₂).
    SpectralConnectivity,

    /// Binary with tiered approximation
    Tiered(ApproximationTier),

    /// Resonator-based O(n log N) approximation
    Resonator,

    /// Auto-select based on topology size
    #[default]
    Auto,
}

impl PhiEngine {
    /// Create a new PhiEngine with specified method
    pub fn new(method: PhiMethod) -> Self {
        Self { method }
    }

    /// Create with auto method selection
    pub fn auto() -> Self {
        Self::new(PhiMethod::Auto)
    }

    /// Get the currently selected method
    pub fn method(&self) -> PhiMethod {
        self.method
    }

    /// Set the calculation method
    pub fn set_method(&mut self, method: PhiMethod) {
        self.method = method;
    }

    /// Suggest best method for a given topology size
    ///
    /// ## Performance-Based Selection (Benchmarked 2026-01-04)
    ///
    /// Based on actual benchmark measurements at HDC_DIMENSION=16,384:
    ///
    /// | Method | Nodes | Time | Recommendation |
    /// |--------|-------|------|----------------|
    /// | Continuous (RealPhi) | 8 | ~2ms | ✅ Fastest |
    /// | Continuous (RealPhi) | 16 | ~4ms | ✅ Fastest |
    /// | Continuous (RealPhi) | 32 | ~35ms | ✅ Best |
    /// | Continuous (RealPhi) | 64 | ~2.8s | ✅ Acceptable |
    /// | Resonator | 8 | ~80ms | ❌ 50x slower |
    /// | Tiered(Exact) | 12 | ~hours | ❌ O(2^n) |
    ///
    /// ## Key Finding
    /// RealPhi (Continuous) is **50x faster** than Resonator for n≤16.
    /// Resonator only becomes beneficial for n>256 due to O(n log n) vs O(n³) scaling.
    pub fn suggest_method(n_nodes: usize) -> PhiMethod {
        match n_nodes {
            // For n≤10, Exact is tractable (2^10 = 1024 partitions, ~10ms)
            // but Continuous is faster anyway, use Continuous
            0..=64 => PhiMethod::SpectralConnectivity,

            // For 65-256, Continuous still works (~2.8s for 64, scales O(n³))
            // At 128 nodes: ~22s, at 256 nodes: ~3min (borderline)
            65..=256 => PhiMethod::SpectralConnectivity,

            // For n>256, only Resonator scales well (O(n log n))
            // Continuous would timeout (256³ = 16M ops, 512³ = 134M ops)
            _ => PhiMethod::Resonator,
        }
    }

    /// Compute Φ for hypervector representations using optimal method
    ///
    /// Automatically selects the best calculation method based on topology size
    /// and available implementations.
    ///
    /// # Arguments
    /// * `node_representations` - Hypervector representations of network nodes
    ///
    /// # Returns
    /// `PhiResult` with the Φ value and metadata
    ///
    /// # Example
    /// ```rust,ignore
    /// let engine = PhiEngine::auto();
    /// let result = engine.compute(&topology.node_representations);
    /// println!("Φ = {:.4}", result.phi);
    /// ```
    pub fn compute(&self, node_representations: &[ContinuousHV]) -> result::PhiResult {
        let n_nodes = node_representations.len();

        // Convert ContinuousHV to ContinuousHV for compatibility
        let real_hvs: Vec<ContinuousHV> = node_representations
            .iter()
            .map(|chv| ContinuousHV::from_vec(chv.values.clone()))
            .collect();

        // Determine effective method
        let effective_method = match self.method {
            PhiMethod::Auto => Self::suggest_method(n_nodes),
            other => other,
        };

        // Calculate Φ based on the effective method
        let (phi_value, method_name): (f64, &'static str) = match effective_method {
            // Spectral connectivity: algebraic connectivity (λ₂) of cosine-similarity graph
            PhiMethod::SpectralConnectivity => {
                let calc = ContinuousPhiCalculator::new();
                let phi = calc.algebraic_connectivity(&real_hvs);
                (phi, "SpectralConnectivity")
            }
            // Tiered binary Φ using ContinuousHV → BinaryHV conversion
            PhiMethod::Tiered(tier) => {
                // Convert ContinuousHV → BinaryHV using the same helper used in topology validation
                let components: Vec<BinaryHV> = real_hvs.iter().map(real_hv_to_hv16).collect();
                let mut calc = TieredPhi::new(tier);
                let phi = calc.compute(&components);
                (phi, "Tiered")
            }
            // Resonator-based Φ on ContinuousHV representations
            PhiMethod::Resonator => {
                let calc = ResonantPhiCalculator::fast();
                let result = calc.compute(&real_hvs);
                (result.phi, "Resonator")
            }
            PhiMethod::Auto => unreachable!("Auto resolved to concrete method above"),
        };

        // Wrap in PhiResult with appropriate category
        let category = match effective_method {
            PhiMethod::SpectralConnectivity => result::PhiCategory::SpectralConnectivity,
            PhiMethod::Tiered(_) => result::PhiCategory::IntegratedInformation,
            PhiMethod::Resonator => result::PhiCategory::SpectralConnectivity,
            PhiMethod::Auto => unreachable!("Auto resolved to concrete method above"),
        };
        result::PhiResult::new(phi_value, method_name, n_nodes).with_category(category)
    }

    /// Get estimated computation time for a given topology size
    ///
    /// Based on benchmark data from 2026-01-04 at HDC_DIMENSION=16,384
    pub fn estimate_time(n_nodes: usize, method: PhiMethod) -> std::time::Duration {
        use std::time::Duration;

        let effective_method = match method {
            PhiMethod::Auto => Self::suggest_method(n_nodes),
            other => other,
        };

        match effective_method {
            PhiMethod::SpectralConnectivity => {
                // O(n³) scaling: 8 nodes = 2ms base
                // Time ≈ 2ms × (n/8)³
                let factor = (n_nodes as f64 / 8.0).powi(3);
                Duration::from_micros((2000.0 * factor) as u64)
            }
            PhiMethod::Resonator => {
                // O(n log n) scaling: 8 nodes = 80ms base
                let factor = (n_nodes as f64 / 8.0) * ((n_nodes as f64).ln() / (8.0_f64).ln());
                Duration::from_millis((80.0 * factor) as u64)
            }
            PhiMethod::Tiered(ApproximationTier::ExhaustivePartition) => {
                // O(2^n) scaling: exponential
                Duration::from_secs(2u64.saturating_pow((n_nodes as u32).saturating_sub(8)))
            }
            #[allow(deprecated)]
            PhiMethod::Tiered(ApproximationTier::SpectralConnectivity) => {
                // Similar to SpectralConnectivity
                let factor = (n_nodes as f64 / 8.0).powi(3);
                Duration::from_micros((3000.0 * factor) as u64)
            }
            PhiMethod::Tiered(_) => {
                // Heuristic/Mock are fast
                Duration::from_micros(100)
            }
            PhiMethod::Auto => unreachable!("Auto resolved to concrete method above"),
        }
    }
}

impl Default for PhiEngine {
    fn default() -> Self {
        Self::auto()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::binary_hv::BinaryHV;
    use crate::hdc::phi_resonant::ResonantPhiCalculator;
    use crate::hdc::phi_topology_validation::real_hv_to_hv16;
    use crate::hdc::unified_hv::ContinuousHV;

    #[test]
    fn test_phi_engine_creation() {
        let engine = PhiEngine::auto();
        assert_eq!(engine.method(), PhiMethod::Auto);
    }

    #[test]
    fn test_method_suggestion() {
        // Based on 2026-01-04 benchmarks: Continuous is fastest for n≤256
        assert!(matches!(
            PhiEngine::suggest_method(8),
            PhiMethod::SpectralConnectivity
        ));
        assert!(matches!(
            PhiEngine::suggest_method(30),
            PhiMethod::SpectralConnectivity
        ));
        assert!(matches!(
            PhiEngine::suggest_method(64),
            PhiMethod::SpectralConnectivity
        ));
        assert!(matches!(
            PhiEngine::suggest_method(100),
            PhiMethod::SpectralConnectivity
        ));
        assert!(matches!(
            PhiEngine::suggest_method(256),
            PhiMethod::SpectralConnectivity
        ));
        // Only for very large topologies, use Resonator (O(n log n))
        assert!(matches!(
            PhiEngine::suggest_method(500),
            PhiMethod::Resonator
        ));
        assert!(matches!(
            PhiEngine::suggest_method(1000),
            PhiMethod::Resonator
        ));
    }

    #[test]
    fn test_time_estimation() {
        // 8 nodes: ~2ms
        let time_8 = PhiEngine::estimate_time(8, PhiMethod::Auto);
        assert!(time_8.as_micros() > 1000 && time_8.as_micros() < 5000);

        // 16 nodes: ~16ms (2 * 2³ = 16ms)
        let time_16 = PhiEngine::estimate_time(16, PhiMethod::Auto);
        assert!(time_16.as_millis() >= 10 && time_16.as_millis() < 30);

        // 64 nodes: ~2s (2 * 8³ = 1024ms)
        let time_64 = PhiEngine::estimate_time(64, PhiMethod::Auto);
        assert!(time_64.as_secs() >= 1 && time_64.as_secs() < 5);
    }

    /// Ensure SpectralConnectivity method uses algebraic connectivity under the hood
    #[test]
    fn test_compute_spectral_connectivity_matches_realphi() {
        // Small dimension for fast tests
        let dim = 128;
        let hvs: Vec<ContinuousHV> = (0..3)
            .map(|i| ContinuousHV::random(dim, 42 + i as u64))
            .collect();

        let engine = PhiEngine::new(PhiMethod::SpectralConnectivity);
        let result = engine.compute(&hvs);

        assert_eq!(result.method, "SpectralConnectivity");
        assert_eq!(result.n_nodes, hvs.len());
        assert!(result.phi >= 0.0 && result.phi <= 1.0);

        // Compare with direct RealPhiCalculator on equivalent ContinuousHV vectors
        let real_hvs: Vec<ContinuousHV> = hvs
            .iter()
            .map(|hv| ContinuousHV::from_vec(hv.values.clone()))
            .collect();
        let calc = ContinuousPhiCalculator::new();
        let direct_phi = calc.algebraic_connectivity(&real_hvs);

        assert!(
            (result.phi - direct_phi).abs() < 1e-9,
            "PhiEngine Continuous φ mismatch: engine={}, direct={}",
            result.phi,
            direct_phi
        );
    }

    /// Ensure Tiered method uses TieredPhi on BinaryHV components
    #[test]
    fn test_compute_tiered_matches_tieredphi() {
        let dim = 128;
        let hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(dim, 100 + i as u64))
            .collect();

        let engine = PhiEngine::new(PhiMethod::Tiered(ApproximationTier::RandomBaseline));
        let result = engine.compute(&hvs);

        assert_eq!(result.method, "Tiered");
        assert_eq!(result.n_nodes, hvs.len());

        // ContinuousHV → BinaryHV conversion must match TieredPhi usage
        let real_hvs: Vec<ContinuousHV> = hvs
            .iter()
            .map(|hv| ContinuousHV::from_vec(hv.values.clone()))
            .collect();
        let components: Vec<BinaryHV> = real_hvs.iter().map(|hv| real_hv_to_hv16(hv)).collect();

        let mut calc = TieredPhi::new(ApproximationTier::RandomBaseline);
        let direct_phi = calc.compute(&components);

        assert!(
            (result.phi - direct_phi).abs() < 1e-9,
            "PhiEngine Tiered φ mismatch: engine={}, direct={}",
            result.phi,
            direct_phi
        );
    }

    /// Ensure Resonator method uses ResonantPhiCalculator::fast
    #[test]
    fn test_compute_resonator_matches_resonant() {
        let dim = 128;
        let hvs: Vec<ContinuousHV> = (0..3)
            .map(|i| ContinuousHV::random(dim, 200 + i as u64))
            .collect();

        let engine = PhiEngine::new(PhiMethod::Resonator);
        let result = engine.compute(&hvs);

        assert_eq!(result.method, "Resonator");
        assert_eq!(result.n_nodes, hvs.len());

        let real_hvs: Vec<ContinuousHV> = hvs
            .iter()
            .map(|hv| ContinuousHV::from_vec(hv.values.clone()))
            .collect();
        let calc = ResonantPhiCalculator::fast();
        let resonant_result = calc.compute(&real_hvs);

        assert!(
            (result.phi - resonant_result.phi).abs() < 1e-9,
            "PhiEngine Resonator φ mismatch: engine={}, direct={}",
            result.phi,
            resonant_result.phi
        );
    }
}
