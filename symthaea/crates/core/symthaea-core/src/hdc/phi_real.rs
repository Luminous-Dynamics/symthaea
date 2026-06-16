// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # DEPRECATED - Use `spectral_connectivity` module instead
//!
//! This module has been renamed to `spectral_connectivity` to accurately
//! reflect what it computes: **algebraic connectivity (λ₂)**, NOT IIT Φ.
//!
//! ## Migration Guide
//!
//! ```rust,ignore
//! // OLD (deprecated):
//! use symthaea::hdc::phi_real::RealPhiCalculator;
//! let phi = calculator.compute(&components);
//!
//! // NEW (correct):
//! use symthaea::hdc::spectral_connectivity::ConnectivityCalculator;
//! let lambda2 = calculator.algebraic_connectivity(&components);
//! ```
//!
//! ## Why This Change?
//!
//! Validation revealed λ₂ has **r = -0.62 correlation** with IIT Φ - they measure
//! nearly opposite properties! See `docs/METRIC_CLARIFICATION.md` for details.
//!
//! ## Valid Use Cases for λ₂
//!
//! ✅ Network connectivity analysis
//! ✅ Synchronization potential estimation
//! ✅ Graph mixing properties
//!
//! ## Invalid Use Cases
//!
//! ❌ Consciousness measurement claims
//! ❌ IIT Φ approximation
//! ❌ Integrated information estimation

#![allow(deprecated)]
#![deprecated(
    since = "0.5.0",
    note = "Module renamed to spectral_connectivity - this measures λ₂, NOT IIT Φ. See docs/METRIC_CLARIFICATION.md"
)]

// Re-export everything from spectral_connectivity for backward compatibility
pub use crate::hdc::spectral_connectivity::*;

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::hdc::unified_hv::ContinuousHV;

    #[test]
    fn test_phi_real_reexports_calculator() {
        let calc = ConnectivityCalculator::new();
        let result = calc.algebraic_connectivity(&[]);
        assert_eq!(result, 0.0, "Empty input should produce 0.0 connectivity");
    }

    #[test]
    fn test_phi_real_algebraic_connectivity_empty() {
        let calc = ConnectivityCalculator::new();
        let result = calc.algebraic_connectivity(&[]);
        assert_eq!(result, 0.0, "Empty components should give 0.0");
    }

    #[test]
    fn test_phi_real_algebraic_connectivity_single() {
        let calc = ConnectivityCalculator::new();
        let components = vec![ContinuousHV::random(256, 42)];
        let result = calc.algebraic_connectivity(&components);
        assert_eq!(result, 0.0, "Single component should give 0.0");
    }

    #[test]
    fn test_phi_real_algebraic_connectivity_two() {
        let calc = ConnectivityCalculator::new();
        let components = vec![ContinuousHV::random(256, 1), ContinuousHV::random(256, 2)];
        let result = calc.algebraic_connectivity(&components);
        assert!(
            result >= 0.0 && result <= 1.0,
            "Lambda2 should be in [0, 1], got {}",
            result
        );
    }

    #[test]
    fn test_phi_real_algebraic_connectivity_three() {
        let calc = ConnectivityCalculator::new();
        let components = vec![
            ContinuousHV::random(256, 1),
            ContinuousHV::random(256, 2),
            ContinuousHV::random(256, 3),
        ];
        let result = calc.algebraic_connectivity(&components);
        assert!(
            result >= 0.0 && result <= 1.0,
            "Lambda2 should be in [0, 1], got {}",
            result
        );
    }

    #[test]
    fn test_phi_real_identical_components() {
        let calc = ConnectivityCalculator::new();
        let hv = ContinuousHV::random(128, 42);
        let components = vec![hv.clone(), hv.clone(), hv.clone()];
        let result = calc.algebraic_connectivity(&components);
        assert!(
            result >= 0.0,
            "Result should be non-negative, got {}",
            result
        );
    }

    #[test]
    fn test_phi_real_similarity_matrix() {
        let calc = ConnectivityCalculator::new();
        let components = vec![
            ContinuousHV::random(128, 1),
            ContinuousHV::random(128, 2),
            ContinuousHV::random(128, 3),
        ];
        let matrix = calc.build_similarity_matrix(&components);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        for i in 0..3 {
            assert!(
                (matrix[i][i] - 1.0).abs() < 1e-10,
                "Diagonal should be 1.0, got {}",
                matrix[i][i]
            );
        }
        for i in 0..3 {
            for j in i + 1..3 {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-10,
                    "Matrix should be symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }
}
