// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Molecular → Continuum coarse-graining bridge.
//!
//! Extracts macroscopic material properties from molecular-level calculations:
//! polarizability → dielectric constant, vibrational modes → heat capacity,
//! orbital gaps → conductivity.
//!
//! References:
//! - Voth, G. A. (2009). *Coarse-Graining of Condensed Phase Systems*.
//! - Clausius-Mossotti relation: ε = (1 + 2Nα/3ε₀)/(1 - Nα/3ε₀)

use std::f64::consts::PI;

/// Clausius-Mossotti: molecular polarizability → bulk dielectric constant.
///
/// ε_r = (1 + 2Nα/(3ε₀)) / (1 - Nα/(3ε₀))
///
/// `alpha` = molecular polarizability (m³ or atomic units)
/// `number_density` = molecules per m³
pub fn clausius_mossotti(alpha_si: f64, number_density: f64) -> f64 {
    let eps_0 = crate::maxwell::EPS_0;
    let chi = number_density * alpha_si / (3.0 * eps_0);
    (1.0 + 2.0 * chi) / (1.0 - chi)
}

/// Molecular polarizability from HOMO-LUMO gap (crude estimate).
/// α ≈ e²/(m_e × ΔE²) × N_electrons (atomic units)
///
/// In SI: α(m³) ≈ 4πε₀ × α(atomic units) × a₀³
pub fn polarizability_from_gap(homo_lumo_gap_hartree: f64, n_electrons: usize) -> f64 {
    if homo_lumo_gap_hartree < 1e-10 {
        return 0.0;
    }
    // In atomic units: α ≈ N / ΔE² (very rough)
    n_electrons as f64 / (homo_lumo_gap_hartree * homo_lumo_gap_hartree)
}

/// Convert polarizability from atomic units (Bohr³) to SI (m³).
pub fn polarizability_au_to_si(alpha_au: f64) -> f64 {
    let a0 = 5.291_772_109e-11; // Bohr radius in meters
    let eps_0 = crate::maxwell::EPS_0;
    4.0 * PI * eps_0 * alpha_au * a0 * a0 * a0
}

/// Heat capacity from vibrational frequencies (Einstein model).
///
/// C_v = Σ_i k_B × (θ_i/T)² × exp(θ_i/T) / (exp(θ_i/T) - 1)²
///
/// where θ_i = ℏω_i / k_B is the Einstein temperature of mode i.
pub fn einstein_heat_capacity(frequencies_hartree: &[f64], temperature: f64) -> f64 {
    let k_b = 3.166_811_563e-6; // Hartree/K
    let mut cv = 0.0;

    for &omega in frequencies_hartree {
        let theta = omega / k_b; // Einstein temperature
        let x = theta / temperature;
        if x < 500.0 {
            // Avoid overflow
            let ex = x.exp();
            cv += k_b * x * x * ex / ((ex - 1.0) * (ex - 1.0));
        }
    }

    cv
}

/// Estimate electrical conductivity from HOMO-LUMO gap.
///
/// For insulators/semiconductors: σ ∝ exp(-E_gap / 2kT)
/// Returns relative conductivity (normalized, not absolute).
pub fn conductivity_from_gap(gap_hartree: f64, temperature: f64) -> f64 {
    let k_b = 3.166_811_563e-6;
    let kt = k_b * temperature;
    (-gap_hartree / (2.0 * kt)).exp()
}

/// Classify material type from HOMO-LUMO gap.
pub fn classify_material(gap_ev: f64) -> &'static str {
    if gap_ev < 0.1 {
        "Metal/Conductor"
    } else if gap_ev < 3.0 {
        "Semiconductor"
    } else {
        "Insulator"
    }
}

/// Thermal de Broglie wavelength: Λ = h/√(2πmkT)
/// Determines when quantum effects become important (Λ ≈ interparticle spacing).
pub fn thermal_de_broglie(mass_kg: f64, temperature: f64) -> f64 {
    let k_b = 1.380_649e-23;
    let h = 6.626_070_15e-34;
    h / (2.0 * PI * mass_kg * k_b * temperature).sqrt()
}

/// Lindemann criterion: melting occurs when RMS displacement ≈ 0.1 × lattice spacing.
/// T_melt ≈ (0.1 × a)² × m × ω_D² / k_B
/// where ω_D is the Debye frequency and a is the lattice constant.
pub fn lindemann_melting_temperature(
    lattice_constant_m: f64,
    mass_kg: f64,
    debye_frequency: f64,
) -> f64 {
    let k_b = 1.380_649e-23;
    let threshold = 0.1 * lattice_constant_m;
    threshold * threshold * mass_kg * debye_frequency * debye_frequency / k_b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clausius_mossotti_vacuum() {
        // Zero polarizability → ε_r = 1 (vacuum)
        let eps = clausius_mossotti(0.0, 1e28);
        assert!((eps - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clausius_mossotti_dilute_gas() {
        // N₂ gas at STP: α = 1.74 × 10⁻⁴⁰ C²·m/J (SI polarizability)
        // n = 2.5 × 10²⁵ m⁻³, expected ε_r ≈ 1.0006
        let alpha_si = 1.74e-40; // SI polarizability
        let n = 2.5e25; // Number density at STP
        let eps = clausius_mossotti(alpha_si, n);
        assert!(
            eps > 1.0 && eps < 1.01,
            "N₂ gas ε_r = {:.6} should be ~1.0006",
            eps
        );
    }

    #[test]
    fn test_conductivity_gap_ordering() {
        // Larger gap → lower conductivity
        let sigma_small = conductivity_from_gap(0.04, 300.0); // ~1 eV
        let sigma_large = conductivity_from_gap(0.15, 300.0); // ~4 eV
        assert!(
            sigma_small > sigma_large,
            "Small gap ({:.2e}) should conduct more than large gap ({:.2e})",
            sigma_small, sigma_large
        );
    }

    #[test]
    fn test_material_classification() {
        assert_eq!(classify_material(0.0), "Metal/Conductor");
        assert_eq!(classify_material(1.1), "Semiconductor");
        assert_eq!(classify_material(5.0), "Insulator");
    }

    #[test]
    fn test_einstein_heat_capacity_high_t() {
        // At high T, C_v → k_B per mode (classical limit)
        let k_b = 3.166_811_563e-6;
        let cv = einstein_heat_capacity(&[0.01], 100_000.0); // Very high T
        assert!(
            (cv - k_b).abs() / k_b < 0.1,
            "Classical limit: C_v={:.2e}, k_B={:.2e}",
            cv, k_b
        );
    }

    #[test]
    fn test_einstein_heat_capacity_low_t() {
        // At low T, C_v → 0 (quantum freeze-out)
        let cv = einstein_heat_capacity(&[0.01], 1.0); // Very low T
        assert!(cv < 1e-20, "Low-T C_v should freeze out: {:.2e}", cv);
    }
}
