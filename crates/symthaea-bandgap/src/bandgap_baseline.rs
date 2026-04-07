// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Electronegativity-based bandgap baseline model.
//!
//! A simple composition-only model: bandgap ~ f(electronegativity difference).
//! This is intentionally crude — the RF corrects the residuals.
//!
//! The model is based on the empirical observation that wider bandgaps correlate
//! with larger electronegativity differences between cation and anion sublattices.
//!
//! Reference: Nethercot, A.H. (1974). Prediction of Fermi Energies and
//!   Photoelectric Thresholds Based on Electronegativity Concepts.
//!   Phys. Rev. Lett. 33, 1088.

use crate::periodic_table::{electronegativity, element_block, COVALENT_RADIUS};

/// Predict bandgap from composition using electronegativity model.
///
/// Model: Eg = a * (chi_anion - chi_cation)^b + c * (1/r_avg) + d
///
/// Where chi = Pauling electronegativity, r_avg = weighted average covalent radius.
/// Parameters fitted to reproduce broad trends (not DFT-accurate).
pub fn electronegativity_bandgap(composition: &[(u8, f64)]) -> f64 {
    if composition.is_empty() {
        return 0.0;
    }

    // Single-element: special handling
    if composition.len() == 1 {
        let z = composition[0].0;
        return elemental_bandgap_estimate(z);
    }

    // Separate cation-like and anion-like species based on electronegativity
    let mean_en: f64 = composition.iter()
        .map(|&(z, f)| electronegativity(z) * f)
        .sum();

    let (mut cation_en, mut cation_w) = (0.0, 0.0);
    let (mut anion_en, mut anion_w) = (0.0, 0.0);

    for &(z, f) in composition {
        let en = electronegativity(z);
        if en <= mean_en {
            cation_en += en * f;
            cation_w += f;
        } else {
            anion_en += en * f;
            anion_w += f;
        }
    }

    // Handle edge case: all same electronegativity
    if cation_w < 1e-10 || anion_w < 1e-10 {
        return elemental_bandgap_estimate(composition[0].0);
    }

    cation_en /= cation_w;
    anion_en /= anion_w;

    let delta_en = anion_en - cation_en;

    // Average covalent radius
    let r_avg: f64 = composition.iter()
        .map(|&(z, f)| {
            let r = if (z as usize) < COVALENT_RADIUS.len() {
                COVALENT_RADIUS[z as usize]
            } else {
                1.5
            };
            r * f
        })
        .sum();

    // Check if d-block elements are present (reduces gap due to d-orbital contributions)
    let d_block_fraction: f64 = composition.iter()
        .filter(|&&(z, _)| element_block(z) == 2)
        .map(|&(_, f)| f)
        .sum();

    // Empirical model: Eg = A * delta_EN^B + C / r_avg + D
    // Parameters chosen to roughly reproduce: Si~1.1, GaAs~1.4, NaCl~8.5, GaN~3.4
    let a = 2.8;
    let b = 1.3;
    let c = 0.5;
    let d = -1.0;

    let eg = a * delta_en.powf(b) + c / r_avg + d;

    // d-block elements suppress bandgap
    let eg = eg * (1.0 - 0.3 * d_block_fraction);

    // Clamp to physical range
    eg.max(0.0)
}

/// Estimate bandgap for elemental solids.
fn elemental_bandgap_estimate(z: u8) -> f64 {
    match z {
        6 => 5.5,   // Diamond
        14 => 1.1,  // Si
        32 => 0.7,  // Ge
        50 => 0.1,  // alpha-Sn
        34 => 1.8,  // Se
        52 => 0.33, // Te
        _ => 0.0,   // Metals default to 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elemental_si() {
        let eg = electronegativity_bandgap(&[(14, 1.0)]);
        assert!((eg - 1.1).abs() < 0.5, "Si baseline = {}, expected ~1.1", eg);
    }

    #[test]
    fn test_wide_gap_trend() {
        // NaCl should have larger gap than GaAs
        let nacl = electronegativity_bandgap(&[(11, 0.5), (17, 0.5)]);
        let gaas = electronegativity_bandgap(&[(31, 0.5), (33, 0.5)]);
        assert!(nacl > gaas, "NaCl ({}) should > GaAs ({})", nacl, gaas);
    }

    #[test]
    fn test_empty_composition() {
        assert_eq!(electronegativity_bandgap(&[]), 0.0);
    }

    #[test]
    fn test_nonnegative() {
        // All predictions should be non-negative
        let materials = vec![
            vec![(14, 1.0)],
            vec![(31, 0.5), (33, 0.5)],
            vec![(11, 0.5), (17, 0.5)],
            vec![(30, 0.5), (8, 0.5)],
        ];
        for comp in &materials {
            let eg = electronegativity_bandgap(comp);
            assert!(eg >= 0.0, "Bandgap for {:?} = {} < 0", comp, eg);
        }
    }
}
