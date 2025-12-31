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
//! | Continuous | O(n³) | High | Research, small networks |
//! | Resonator | O(n log n) | Medium | Large networks, real-time |
//! | Tiered | O(1) to O(2ⁿ) | Variable | Testing to exact |
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

mod calculator;
mod result;

// Re-export main types
pub use calculator::{PhiCalculator, Complexity};
pub use result::{PhiResult, PhiUncertainty};

// Re-export specific implementations (from existing hdc module)
pub use crate::hdc::phi_real::RealPhiCalculator as ContinuousPhiCalculator;
pub use crate::hdc::tiered_phi::{TieredPhi, ApproximationTier, TieredPhiConfig};
// Re-export resonator when available:
// pub use crate::hdc::phi_resonant::ResonatorPhi;

/// PhiEngine facade for easy method selection and comparison
#[derive(Clone)]
pub struct PhiEngine {
    /// Currently selected method
    method: PhiMethod,
}

/// Available Φ calculation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiMethod {
    /// Continuous (RealHV-based, cosine similarity)
    Continuous,

    /// Binary with tiered approximation
    Tiered(ApproximationTier),

    /// Resonator-based O(n log N) approximation
    Resonator,

    /// Auto-select based on topology size
    Auto,
}

impl Default for PhiMethod {
    fn default() -> Self {
        PhiMethod::Auto
    }
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
    pub fn suggest_method(n_nodes: usize) -> PhiMethod {
        match n_nodes {
            0..=12 => PhiMethod::Tiered(ApproximationTier::Exact),
            13..=50 => PhiMethod::Continuous,
            51..=500 => PhiMethod::Tiered(ApproximationTier::Spectral),
            _ => PhiMethod::Resonator,
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

    #[test]
    fn test_phi_engine_creation() {
        let engine = PhiEngine::auto();
        assert_eq!(engine.method(), PhiMethod::Auto);
    }

    #[test]
    fn test_method_suggestion() {
        assert!(matches!(PhiEngine::suggest_method(8), PhiMethod::Tiered(ApproximationTier::Exact)));
        assert!(matches!(PhiEngine::suggest_method(30), PhiMethod::Continuous));
        assert!(matches!(PhiEngine::suggest_method(100), PhiMethod::Tiered(ApproximationTier::Spectral)));
        assert!(matches!(PhiEngine::suggest_method(1000), PhiMethod::Resonator));
    }
}
