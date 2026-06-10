// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compound Stability Predictor
//!
//! Predicts thermodynamic stability of compounds from elemental binding
//! energies and mixing entropy. Connects nuclear physics (SEMF binding)
//! to materials science (HDC similarity search).

use serde::{Deserialize, Serialize};

/// Stability prediction for a compound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityPrediction {
    /// Estimated formation energy (eV/atom, negative = stable)
    pub formation_energy: f64,
    /// Confidence in the prediction (0-1)
    pub confidence: f64,
    /// Whether the compound is predicted to be thermodynamically stable
    pub is_stable: bool,
    /// Mixing entropy contribution (eV/atom)
    pub mixing_entropy: f64,
    /// Elements involved
    pub elements: Vec<(u16, f64)>, // (Z, mole fraction)
    /// Human-readable formula
    pub formula: String,
}

/// Predict compound stability from elemental composition.
///
/// Uses a simplified model:
/// - Formation energy estimated from elemental cohesive energies
/// - Mixing entropy from ideal solution model: -kT Σ x_i ln(x_i)
/// - Stability = formation_energy + T * mixing_entropy < 0
///
/// This is a first-order approximation — real DFT calculations are
/// needed for accurate predictions, but this provides a useful screening.
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

    // Approximate formation energy from elemental cohesive energies
    // Negative = exothermic formation = stable
    // This uses a simplified Miedema-like model
    let formation_energy = estimate_formation_energy(elements);

    // Total free energy
    let free_energy = formation_energy - temperature_k * mixing_entropy;
    let is_stable = if elements.len() < 2 {
        true // Pure elements are stable by definition
    } else {
        free_energy < 0.0
    };

    // Confidence based on how many elements and how well-characterized
    let confidence = if elements.len() <= 3 {
        0.6 // Binary/ternary: reasonable confidence
    } else {
        0.3 // Complex: lower confidence
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

/// Simplified formation energy estimate (Miedema-inspired).
///
/// Uses electronegativity difference and atomic volume mismatch
/// as proxies for formation energy.
fn estimate_formation_energy(elements: &[(u16, f64)]) -> f64 {
    if elements.len() < 2 {
        return 0.0; // Pure element — no formation
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
            _ => 1.5, // Default estimate
        }
    };

    // Weighted average electronegativity
    let avg_en: f64 = elements.iter().map(|(z, x)| x * electroneg(*z)).sum();

    // Formation energy ∝ -Σ x_i (χ_i - χ_avg)²
    // Negative when electronegativities differ (ionic/polar bonding)
    let delta_en_sq: f64 = elements
        .iter()
        .map(|(z, x)| x * (electroneg(*z) - avg_en).powi(2))
        .sum();

    // Scale factor (empirical, ~-1 eV per unit χ² difference)
    -delta_en_sq * 1.0
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
    fn test_pure_element_zero_formation() {
        let pred = predict_stability(&[(26, 1.0)], 300.0);
        assert!(
            pred.formation_energy.abs() < 0.01,
            "Pure Fe should have ~0 formation energy: {}",
            pred.formation_energy
        );
    }

    #[test]
    fn test_binary_compound_negative_formation() {
        // NaCl-like: large electronegativity difference → negative formation energy
        let pred = predict_stability(&[(11, 0.5), (17, 0.5)], 300.0);
        assert!(
            pred.formation_energy < 0.0,
            "NaCl should have negative formation energy: {}",
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
