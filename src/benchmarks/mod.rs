//! Benchmarks module - performance and validation benchmarks

pub mod causal_cantor;
pub mod causal_tower;
pub mod cognitive_loop_validation;
pub mod prediction_gate;
pub mod primitive_learning_cognitive;
pub mod primitive_learning;
pub mod unified_architecture;
pub mod fep_temporal_benchmark;
pub mod unified_causal_understanding;

// Re-exports for benchmark modules - use types from hdc (which re-exports from symthaea_core)
pub use crate::hdc::CausalDirection;
pub use crate::hdc::{CausalDiscoveryResult, CausalFeatures};

/// Stub module for tuebingen adapter (information theoretic causal discovery)
pub mod tuebingen_adapter {
    pub use super::{CausalDirection, CausalDiscoveryResult, CausalFeatures};

    /// Discover causal direction using information theoretic methods (stub)
    pub fn discover_by_information_theoretic(x: &[f64], y: &[f64]) -> CausalDiscoveryResult {
        // Stub implementation - uses entropy-based heuristic
        let x_entropy = entropy(x);
        let y_entropy = entropy(y);

        // Calculate p_forward based on entropy difference
        let p_forward = if x_entropy + y_entropy > 0.0 {
            // Lower entropy in X suggests X is cause (more structured)
            0.5 + 0.3 * (y_entropy - x_entropy) / (x_entropy + y_entropy + 1e-10)
        } else {
            0.5
        };
        let p_backward = 1.0 - p_forward;

        let direction = if p_forward > 0.5 {
            CausalDirection::Forward
        } else {
            CausalDirection::Backward
        };

        CausalDiscoveryResult {
            direction,
            p_forward,
            p_backward,
            confidence: (p_forward - 0.5).abs() * 2.0, // 0 to 1 scale
            features: CausalFeatures {
                reci_score: 0.0,
                igci_score: 0.0,
                anm_score: 0.0,
                higher_order_score: 0.0,
            },
        }
    }

    fn entropy(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        // Approximate entropy using variance (higher variance = higher entropy)
        variance.ln().max(0.0)
    }
}
