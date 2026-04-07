// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Feature extraction for bandgap ML prediction.
//!
//! Extracts composition-based features from (elements, fractions, crystal system).
//! Features are designed to capture the physics that determines bandgap:
//! electronegativity difference, ionic character, orbital character, size effects.

use crate::periodic_table::*;
use crate::training_data::CrystalSystem;

/// Number of features in the feature vector.
pub const N_FEATURES: usize = 16;

/// Extract physics-motivated features from composition and crystal system.
///
/// Features (16 total):
///  0: weighted mean electronegativity
///  1: electronegativity difference (anion - cation sublattice)
///  2: electronegativity std dev
///  3: weighted mean covalent radius (Angstrom)
///  4: radius ratio (smallest / largest element)
///  5: weighted mean atomic number
///  6: max atomic number
///  7: s-block fraction
///  8: p-block fraction
///  9: d-block fraction
/// 10: f-block fraction
/// 11: number of distinct elements
/// 12: crystal system (encoded 0-6)
/// 13: max period
/// 14: mean period
/// 15: ionicity estimate (delta_EN / sum_EN)
pub fn extract_features(composition: &[(u8, f64)], crystal: &CrystalSystem) -> [f64; N_FEATURES] {
    let mut features = [0.0f64; N_FEATURES];

    if composition.is_empty() {
        return features;
    }

    // Weighted mean electronegativity
    let mean_en: f64 = composition.iter()
        .map(|&(z, f)| electronegativity(z) * f)
        .sum();
    features[0] = mean_en;

    // Electronegativity difference: separate cation/anion sublattices
    let (mut cat_en, mut cat_w) = (0.0, 0.0);
    let (mut ani_en, mut ani_w) = (0.0, 0.0);
    for &(z, f) in composition {
        let en = electronegativity(z);
        if en <= mean_en {
            cat_en += en * f;
            cat_w += f;
        } else {
            ani_en += en * f;
            ani_w += f;
        }
    }
    let delta_en = if cat_w > 1e-10 && ani_w > 1e-10 {
        ani_en / ani_w - cat_en / cat_w
    } else {
        0.0
    };
    features[1] = delta_en;

    // EN std dev
    let en_var: f64 = composition.iter()
        .map(|&(z, f)| {
            let en = electronegativity(z);
            (en - mean_en).powi(2) * f
        })
        .sum();
    features[2] = en_var.sqrt();

    // Weighted mean covalent radius
    let mean_r: f64 = composition.iter()
        .map(|&(z, f)| {
            let r = if (z as usize) < COVALENT_RADIUS.len() {
                COVALENT_RADIUS[z as usize]
            } else {
                1.5
            };
            r * f
        })
        .sum();
    features[3] = mean_r;

    // Radius ratio
    let radii: Vec<f64> = composition.iter()
        .map(|&(z, _)| {
            if (z as usize) < COVALENT_RADIUS.len() {
                COVALENT_RADIUS[z as usize]
            } else {
                1.5
            }
        })
        .collect();
    let r_min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
    let r_max = radii.iter().cloned().fold(0.0f64, f64::max);
    features[4] = if r_max > 1e-10 { r_min / r_max } else { 1.0 };

    // Weighted mean Z
    let mean_z: f64 = composition.iter()
        .map(|&(z, f)| z as f64 * f)
        .sum();
    features[5] = mean_z;

    // Max Z
    features[6] = composition.iter().map(|&(z, _)| z as f64).fold(0.0f64, f64::max);

    // Block fractions
    let (mut s_f, mut p_f, mut d_f, mut f_f) = (0.0, 0.0, 0.0, 0.0);
    for &(z, f) in composition {
        match element_block(z) {
            0 => s_f += f,
            1 => p_f += f,
            2 => d_f += f,
            3 => f_f += f,
            _ => {}
        }
    }
    features[7] = s_f;
    features[8] = p_f;
    features[9] = d_f;
    features[10] = f_f;

    // Number of distinct elements
    features[11] = composition.len() as f64;

    // Crystal system encoding
    features[12] = match crystal {
        CrystalSystem::Cubic => 0.0,
        CrystalSystem::Hexagonal => 1.0,
        CrystalSystem::Tetragonal => 2.0,
        CrystalSystem::Orthorhombic => 3.0,
        CrystalSystem::Monoclinic => 4.0,
        CrystalSystem::Rhombohedral => 5.0,
        CrystalSystem::Triclinic => 6.0,
    };

    // Max and mean period
    let max_period = composition.iter()
        .map(|&(z, _)| element_period(z) as f64)
        .fold(0.0f64, f64::max);
    let mean_period: f64 = composition.iter()
        .map(|&(z, f)| element_period(z) as f64 * f)
        .sum();
    features[13] = max_period;
    features[14] = mean_period;

    // Ionicity estimate
    let sum_en: f64 = composition.iter()
        .map(|&(z, _)| electronegativity(z))
        .sum::<f64>()
        .max(1e-10);
    features[15] = delta_en / sum_en * composition.len() as f64;

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count() {
        let f = extract_features(&[(14, 1.0)], &CrystalSystem::Cubic);
        assert_eq!(f.len(), N_FEATURES);
    }

    #[test]
    fn test_si_features() {
        let f = extract_features(&[(14, 1.0)], &CrystalSystem::Cubic);
        assert!((f[0] - 1.9).abs() < 0.01, "Si EN = {}", f[0]);
        assert!((f[1]).abs() < 0.01, "Si delta_EN = {} (should be ~0)", f[1]);
        assert!((f[5] - 14.0).abs() < 0.01, "Si mean Z = {}", f[5]);
        assert_eq!(f[12], 0.0); // Cubic
    }

    #[test]
    fn test_gaas_features() {
        let f = extract_features(&[(31, 0.5), (33, 0.5)], &CrystalSystem::Cubic);
        assert!(f[1] > 0.0, "GaAs should have positive delta_EN");
        assert_eq!(f[11], 2.0); // 2 elements
    }

    #[test]
    fn test_empty() {
        let f = extract_features(&[], &CrystalSystem::Cubic);
        assert!(f.iter().all(|&v| v == 0.0));
    }
}
