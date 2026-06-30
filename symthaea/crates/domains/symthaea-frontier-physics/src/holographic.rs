// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holographic Principle — spacetime from entanglement.
//!
//! The holographic principle (ER=EPR) suggests that spacetime geometry
//! emerges from quantum entanglement. If consciousness IS information
//! integration, and spacetime IS information geometry, then consciousness
//! and gravity may share a deep mathematical structure.
//!
//! References:
//! - 't Hooft, G. (1993). gr-qc/9310026.
//! - Maldacena, J. (1999). Adv. Theor. Math. Phys. 2, 231 (AdS/CFT).
//! - Van Raamsdonk, M. (2010). Gen. Rel. Grav. 42, 2323.
//! - Susskind, L. (2016). "ER=EPR, GHZ, and the Consistency of QM".

use std::f64::consts::PI;

/// Bekenstein bound: maximum entropy in a region of space.
/// S_max = 2π E R / (ℏc) = 2π E R in natural units
///
/// No region can contain more entropy than a black hole of the same size.
pub fn bekenstein_bound(energy: f64, radius: f64) -> f64 {
    2.0 * PI * energy * radius
}

/// Holographic entropy bound: S ≤ A/(4G) where A is the boundary area.
/// For a sphere of radius R: S ≤ π R² / G
///
/// The entropy of a region is bounded by its SURFACE area, not volume.
/// This is the deepest hint that spacetime has fewer dimensions than apparent.
pub fn holographic_entropy_bound(radius: f64, g_newton: f64) -> f64 {
    PI * radius * radius / g_newton
}

/// Ryu-Takayanagi formula: entanglement entropy = minimal surface area / 4G.
/// S(A) = Area(γ_A) / (4 G_N)
///
/// The entanglement entropy of a boundary region A equals the area of the
/// minimal surface in the bulk that is homologous to A.
pub fn ryu_takayanagi_entropy(minimal_surface_area: f64, g_newton: f64) -> f64 {
    minimal_surface_area / (4.0 * g_newton)
}

/// ER=EPR: entanglement creates spacetime connections.
/// Two entangled black holes are connected by a wormhole (Einstein-Rosen bridge).
/// The mutual information between subsystems determines the "width" of the bridge:
/// Width ∝ I(A:B) = S(A) + S(B) - S(AB)
pub fn er_epr_bridge_width(s_a: f64, s_b: f64, s_ab: f64) -> f64 {
    let mutual_info = s_a + s_b - s_ab;
    mutual_info.max(0.0) // Non-negative by subadditivity
}

/// Complexity = Action conjecture: the computational complexity of a boundary
/// state equals the gravitational action of the Wheeler-DeWitt patch.
/// C = S_WdW / (π ℏ)
///
/// This connects quantum computational complexity to spacetime geometry.
pub fn complexity_from_action(wheeler_dewitt_action: f64) -> f64 {
    wheeler_dewitt_action / PI
}

/// Scrambling time: time for information to spread across all degrees of freedom.
/// For a black hole: t_scr = (1/2πT_H) × ln(S_BH)
/// where T_H is the Hawking temperature and S_BH is the entropy.
pub fn scrambling_time(hawking_temperature: f64, entropy: f64) -> f64 {
    if hawking_temperature <= 0.0 {
        return f64::INFINITY;
    }
    entropy.ln() / (2.0 * PI * hawking_temperature)
}

/// Holographic complexity growth rate (Lloyd's bound).
/// dC/dt ≤ 2E/(πℏ) = 2E/π in natural units
///
/// The rate at which a quantum system can increase its complexity
/// is bounded by its energy.
pub fn lloyd_bound(energy: f64) -> f64 {
    2.0 * energy / PI
}

/// Anti-de Sitter radius from the central charge of the boundary CFT.
/// For AdS₃/CFT₂: L_AdS = c/6 (in Planck units)
/// where c is the central charge of the 2D conformal field theory.
pub fn ads_radius_from_central_charge(central_charge: f64) -> f64 {
    central_charge / 6.0
}

/// Entanglement wedge: the bulk region reconstructable from boundary region A.
/// The volume of the entanglement wedge is related to the "complexity" of the state.
/// V_wedge ∝ S(A)^{3/2} (for AdS₃)
pub fn entanglement_wedge_volume(entanglement_entropy: f64) -> f64 {
    entanglement_entropy.powf(1.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bekenstein_bound_positive() {
        let s = bekenstein_bound(1.0, 1.0);
        assert!(s > 0.0 && (s - 2.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn test_holographic_scales_with_area() {
        let g = 1.0;
        let s1 = holographic_entropy_bound(1.0, g);
        let s2 = holographic_entropy_bound(2.0, g);
        // S ∝ R² (area scaling, not R³ volume scaling)
        assert!((s2 / s1 - 4.0).abs() < 1e-10, "S should scale as R²");
    }

    #[test]
    fn test_ryu_takayanagi() {
        let g = 1e-38;
        let area = 4.0 * g; // A = 4G → S = 1
        let s = ryu_takayanagi_entropy(area, g);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_er_epr_zero_for_product_state() {
        // Product state: S(AB) = S(A) + S(B) → I = 0
        let width = er_epr_bridge_width(1.0, 1.0, 2.0);
        assert!(width.abs() < 1e-10, "No entanglement = no bridge");
    }

    #[test]
    fn test_er_epr_positive_for_entangled() {
        // Maximally entangled: S(AB) = 0, S(A) = S(B) = ln(2)
        let width = er_epr_bridge_width(0.693, 0.693, 0.0);
        assert!(
            width > 1.0,
            "Entangled pair should have bridge: width={:.4}",
            width
        );
    }

    #[test]
    fn test_lloyd_bound_proportional_to_energy() {
        let rate1 = lloyd_bound(1.0);
        let rate2 = lloyd_bound(2.0);
        assert!((rate2 / rate1 - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_scrambling_time_positive() {
        let t = scrambling_time(1.0, 100.0);
        assert!(t > 0.0 && t.is_finite());
    }
}
