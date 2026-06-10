// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Topological Quantum Field Theory — topology of consciousness.
//!
//! Topological invariants are robust against local perturbations,
//! making them candidates for the "hard problem" of consciousness:
//! what physical property is preserved under continuous deformation?
//!
//! References:
//! - Witten, E. (1988). Commun. Math. Phys. 117, 353 (Chern-Simons).
//! - Kitaev, A. & Preskill, J. (2006). Phys. Rev. Lett. 96, 110404.
//! - Levin, M. & Wen, X.-G. (2006). Phys. Rev. Lett. 96, 110405.

use std::f64::consts::PI;

/// Topological entanglement entropy: S_topo = -ln(D)
/// where D is the total quantum dimension of the topological order.
///
/// D = √(Σ d_i²) where d_i are the quantum dimensions of anyonic excitations.
pub fn topological_entanglement_entropy(quantum_dimensions: &[f64]) -> f64 {
    let d_total_sq: f64 = quantum_dimensions.iter().map(|d| d * d).sum();
    let d_total = d_total_sq.sqrt();
    -(d_total.ln())
}

/// Chern number: topological invariant of a 2D band structure.
/// C = (1/2π) ∫ F dkx dky  where F is the Berry curvature.
///
/// For a two-level system H(k) = d(k)·σ:
/// C = (1/4π) ∫ d̂ · (∂d̂/∂kx × ∂d̂/∂ky) dkx dky
///
/// Returns the Chern number for a given Berry curvature field.
pub fn chern_number(berry_curvature: &[f64], dkx: f64, dky: f64, nx: usize, ny: usize) -> f64 {
    let mut integral = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            integral += berry_curvature[i * ny + j] * dkx * dky;
        }
    }
    integral / (2.0 * PI)
}

/// Berry phase around a closed loop: γ = -Im ∮ ⟨ψ|∇_k ψ⟩ · dk
///
/// For a two-level system with a conical intersection:
/// γ = π × (winding number around the degeneracy)
pub fn berry_phase_two_level(winding_number: i32) -> f64 {
    PI * winding_number as f64
}

/// Jones polynomial evaluation at t for a trefoil knot.
/// V_trefoil(t) = -t⁻⁴ + t⁻³ + t⁻¹
///
/// Knot invariants classify topological states — different knots
/// correspond to different types of anyonic excitations.
pub fn jones_polynomial_trefoil(t: f64) -> f64 {
    -t.powi(-4) + t.powi(-3) + t.powi(-1)
}

/// Jones polynomial for the unknot: V(t) = 1
pub fn jones_polynomial_unknot(_t: f64) -> f64 {
    1.0
}

/// Toric code ground state degeneracy on a torus: 4-fold degenerate.
/// This is the simplest topological quantum error-correcting code.
pub fn toric_code_degeneracy() -> usize {
    4
}

/// Anyonic fusion rules: a × b = Σ N_{ab}^c c
/// For Fibonacci anyons: τ × τ = 1 + τ
/// Returns the fusion matrix N for the Fibonacci model.
pub fn fibonacci_fusion_rules() -> [[u32; 2]; 2] {
    // States: {1, τ}
    // 1 × 1 = 1, 1 × τ = τ, τ × 1 = τ, τ × τ = 1 + τ
    [
        [1, 0], // 1 × 1 → (N^1, N^τ) = (1, 0)
        [1, 1], // τ × τ → (N^1, N^τ) = (1, 1)
    ]
}

/// Golden ratio: the quantum dimension of the Fibonacci anyon τ.
/// d_τ = φ = (1 + √5)/2
pub fn fibonacci_quantum_dimension() -> f64 {
    (1.0 + 5.0_f64.sqrt()) / 2.0
}

/// Topological complexity: information-theoretic measure of topological order.
/// For a system with anyons of quantum dimensions d_i:
/// Complexity = ln(D) = ½ ln(Σ d_i²)
pub fn topological_complexity(quantum_dimensions: &[f64]) -> f64 {
    let d_total_sq: f64 = quantum_dimensions.iter().map(|d| d * d).sum();
    0.5 * d_total_sq.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toric_code_entropy() {
        // Toric code: D = 2, S_topo = -ln(2)
        let s = topological_entanglement_entropy(&[1.0, 1.0, 1.0, 1.0]); // 4 anyons of d=1
        let expected = -(4.0_f64.sqrt().ln()); // -ln(2)
        assert!((s - expected).abs() < 1e-14);
    }

    #[test]
    fn test_fibonacci_golden_ratio() {
        let phi = fibonacci_quantum_dimension();
        assert!((phi - 1.618_033_988).abs() < 1e-6, "φ = {:.6}", phi);
    }

    #[test]
    fn test_fibonacci_fusion_tau_squared() {
        let rules = fibonacci_fusion_rules();
        // τ × τ = 1 + τ
        assert_eq!(rules[1][0], 1); // Contains 1
        assert_eq!(rules[1][1], 1); // Contains τ
    }

    #[test]
    fn test_jones_trefoil_at_one() {
        // V_trefoil(1) = -1 + 1 + 1 = 1
        let v = jones_polynomial_trefoil(1.0);
        assert!((v - 1.0).abs() < 1e-14, "V_trefoil(1) = {:.6}", v);
    }

    #[test]
    fn test_jones_unknot() {
        assert_eq!(jones_polynomial_unknot(0.5), 1.0);
    }

    #[test]
    fn test_berry_phase_quantized() {
        // Berry phase should be integer multiples of π
        let gamma = berry_phase_two_level(1);
        assert!((gamma - PI).abs() < 1e-14);
    }

    #[test]
    fn test_chern_number_trivial() {
        // Zero Berry curvature → Chern number = 0
        let bc = vec![0.0; 100];
        let c = chern_number(&bc, 0.1, 0.1, 10, 10);
        assert!(c.abs() < 1e-14);
    }
}
