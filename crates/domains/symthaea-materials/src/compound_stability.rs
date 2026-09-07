// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advisory compound-composition stability screening.
//!
//! This module provides a cheap composition heuristic from electronegativity
//! variance and ideal mixing entropy.  It is useful for ranking/exploration, but
//! it is **not** thermodynamic phase-stability evidence.  In particular, the
//! current formation-energy proxy is `-weighted_variance(electronegativity)`, so
//! a multicomponent composition is structurally biased toward a non-positive
//! proxy.  No crystal structure, competing phases/convex hull, periodic electronic
//! relaxation, phonons, or synthesis pathway is evaluated here.
//!
//! Callers must therefore treat `formation_energy`, `confidence`, and `is_stable`
//! as advisory screening outputs only.  The Matter Observatory crystal/phase
//! admission boundary intentionally does not accept them as a substitute for
//! explicit thermodynamic phase-competition evidence.

use serde::{Deserialize, Serialize};

/// Advisory screening result for a compound composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityPrediction {
    /// Heuristic formation-energy proxy (eV/atom).
    ///
    /// This is not a DFT formation enthalpy/free energy and must not be used as
    /// energy-above-hull or convex-hull evidence.
    pub formation_energy: f64,
    /// Fixed heuristic score based primarily on composition cardinality.
    ///
    /// This is not a calibrated probability or statistical confidence interval.
    pub confidence: f64,
    /// Advisory screening flag from this simplified composition model.
    ///
    /// `true` does not establish thermodynamic phase stability, dynamical
    /// stability, existence of a crystal structure, or synthesizability.
    pub is_stable: bool,
    /// Ideal-mixing entropy contribution used by the screening heuristic (eV/atom).
    pub mixing_entropy: f64,
    /// Elements involved.
    pub elements: Vec<(u16, f64)>, // (Z, mole fraction)
    /// Human-readable formula.
    pub formula: String,
}

/// Produce an advisory stability *screen* from elemental composition.
///
/// Uses a simplified model:
/// - formation-energy proxy from electronegativity variance;
/// - ideal-solution mixing entropy: `-kT Σ x_i ln(x_i)`;
/// - advisory flag from the resulting simplified free-energy proxy.
///
/// This function does not evaluate a crystal structure or competing phases and
/// cannot establish thermodynamic phase stability.  Use a periodic, relaxed,
/// phase-competition workflow (and dynamical stability where applicable) before
/// promoting a candidate beyond heuristic screening.
pub fn predict_stability(
    elements: &[(u16, f64)], // (Z, mole_fraction)
    temperature_k: f64,
) -> StabilityPrediction {
    let k_b_ev = 8.617e-5; // Boltzmann constant in eV/K

    // Mixing entropy (ideal solution)
    let mixing_entropy: f64 = -k_b_ev
        * elements
            .iter()
            .filter(|(_, x)| *x > 0.0 && *x < 1.0)
            .map(|(_, x)| x * x.ln())
            .sum::<f64>();

    // Advisory formation-energy proxy from electronegativity variance.
    // This is not a crystal/DFT formation energy.
    let formation_energy = estimate_formation_energy(elements);

    // Simplified screening free-energy proxy.
    let free_energy = formation_energy - temperature_k * mixing_entropy;
    let is_stable = if elements.len() < 2 {
        true // Pure-element screening convention only.
    } else {
        free_energy < 0.0
    };

    // Heuristic score; not calibrated probability/confidence.
    let confidence = if elements.len() <= 3 {
        0.6
    } else {
        0.3
    };

    let formula = if elements.is_empty() {
        "Pure Preset".to_string()
    } else {
        elements
            .iter()
            .map(|(z, x)| {
                let sym = element_symbol(*z);
                if (*x - 1.0).abs() < 0.01 {
                    sym.to_string()
                } else {
                    format!("{}{:.1}", sym, x)
                }
            })
            .collect::<Vec<_>>()
            .join("")
    };

    StabilityPrediction {
        formation_energy,
        confidence,
        is_stable,
        mixing_entropy,
        elements: elements.to_vec(),
        formula,
    }
}

/// Simplified composition proxy inspired by electronegativity-mismatch models.
///
/// The current expression is the negative weighted variance of electronegativity,
/// so it is non-positive for ordinary non-negative mole fractions.  That property
/// is precisely why this value is advisory and cannot act as a thermodynamic
/// phase-stability gate.
fn estimate_formation_energy(elements: &[(u16, f64)]) -> f64 {
    if elements.len() < 2 {
        return 0.0;
    }

    // Approximate electronegativity (Pauling scale, simplified)
    let electroneg = |z: u16| -> f64 {
        match z {
            1 => 2.20,
            2 => 0.0,
            3 => 0.98,
            4 => 1.57,
            5 => 2.04,
            6 => 2.55,
            7 => 3.04,
            8 => 3.44,
            11 => 0.93,
            12 => 1.31,
            13 => 1.61,
            14 => 1.90,
            15 => 2.19,
            16 => 2.58,
            17 => 3.16,
            19 => 0.82,
            20 => 1.00,
            22 => 1.54,
            24 => 1.66,
            25 => 1.55,
            26 => 1.83,
            27 => 1.88,
            28 => 1.91,
            29 => 1.90,
            30 => 1.65,
            32 => 2.01,
            33 => 2.18,
            34 => 2.55,
            42 => 2.16,
            47 => 1.93,
            48 => 1.69,
            50 => 1.96,
            56 => 0.89,
            74 => 2.36,
            78 => 2.28,
            79 => 2.54,
            82 => 2.33,
            83 => 2.02,
            92 => 1.38,
            _ => 1.5,
        }
    };

    let avg_en: f64 = elements.iter().map(|(z, x)| x * electroneg(*z)).sum();

    // Negative weighted variance.  This cannot establish a physical formation
    // energy or phase stability; see the module-level documentation.
    let delta_en_sq: f64 = elements
        .iter()
        .map(|(z, x)| x * (electroneg(*z) - avg_en).powi(2))
        .sum();

    -delta_en_sq
}

fn element_symbol(z: u16) -> &'static str {
    match z {
        1 => "H",
        2 => "He",
        3 => "Li",
        4 => "Be",
        5 => "B",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        10 => "Ne",
        11 => "Na",
        12 => "Mg",
        13 => "Al",
        14 => "Si",
        15 => "P",
        16 => "S",
        17 => "Cl",
        20 => "Ca",
        22 => "Ti",
        24 => "Cr",
        26 => "Fe",
        28 => "Ni",
        29 => "Cu",
        30 => "Zn",
        47 => "Ag",
        48 => "Cd",
        50 => "Sn",
        74 => "W",
        78 => "Pt",
        79 => "Au",
        82 => "Pb",
        83 => "Bi",
        92 => "U",
        _ => "X",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_element_zero_formation_proxy() {
        let pred = predict_stability(&[(26, 1.0)], 300.0);
        assert!(
            pred.formation_energy.abs() < 0.01,
            "Pure Fe should have ~0 formation proxy: {}",
            pred.formation_energy
        );
    }

    #[test]
    fn test_binary_compound_negative_screening_proxy() {
        let pred = predict_stability(&[(11, 0.5), (17, 0.5)], 300.0);
        assert!(
            pred.formation_energy < 0.0,
            "NaCl-like composition should have a negative heuristic proxy: {}",
            pred.formation_energy
        );
        assert!(pred.is_stable);
    }

    #[test]
    fn test_mixing_entropy_positive() {
        let pred = predict_stability(&[(26, 0.5), (28, 0.5)], 300.0);
        assert!(
            pred.mixing_entropy > 0.0,
            "Mixing entropy should be positive: {}",
            pred.mixing_entropy
        );
    }

    #[test]
    fn test_formula_generation() {
        let pred = predict_stability(&[(26, 0.5), (28, 0.5)], 300.0);
        assert!(
            pred.formula.contains("Fe") && pred.formula.contains("Ni"),
            "Formula should contain Fe and Ni: {}",
            pred.formula
        );
    }
}
