// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Feature extraction for bandgap ML prediction.
//!
//! Extracts 18 composition-based features from (elements, fractions, crystal system).
//! Features capture the physics that determines bandgap: electronegativity
//! difference, ionic character, orbital character, size effects, and
//! Pettifor chemical scale.
//!
//! Analogous to the 12-dimensional nuclear feature vector in symthaea-nuclear.

use crate::bandgap_baseline::electronegativity_bandgap;
use crate::periodic_table::*;
use crate::training_data::CrystalSystem;

/// Number of features in the feature vector.
pub const N_FEATURES: usize = 18;

/// Extract physics-motivated features from composition and crystal system.
///
/// Features (18 total):
///  0: weighted mean electronegativity
///  1: electronegativity range (max - min across elements)
///  2: electronegativity standard deviation
///  3: weighted mean atomic radius (Angstrom)
///  4: atomic radius range (max - min)
///  5: weighted mean ionization energy (eV)
///  6: weighted mean electron affinity (eV)
///  7: weighted mean atomic mass (amu)
///  8: total valence electrons per atom
///  9: number of distinct elements
/// 10: weighted mean period
/// 11: weighted mean group (0 for d/f-block)
/// 12: fraction of d-block elements
/// 13: fraction of p-block elements
/// 14: max oxidation state estimate
/// 15: electronegativity model baseline bandgap
/// 16: crystal system ordinal (0-7)
/// 17: Pettifor chemical scale mean
pub fn extract_features(composition: &[(u8, f64)], crystal: &CrystalSystem) -> [f64; N_FEATURES] {
    let mut features = [0.0f64; N_FEATURES];

    if composition.is_empty() {
        return features;
    }

    // ── 0: Weighted mean electronegativity ──
    let mean_en: f64 = composition
        .iter()
        .map(|&(z, f)| electronegativity(z) * f)
        .sum();
    features[0] = mean_en;

    // ── 1: Electronegativity range (max - min) ──
    let en_values: Vec<f64> = composition
        .iter()
        .map(|&(z, _)| electronegativity(z))
        .collect();
    let en_max = en_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let en_min = en_values.iter().cloned().fold(f64::INFINITY, f64::min);
    features[1] = en_max - en_min;

    // ── 2: Electronegativity std dev ──
    let en_var: f64 = composition
        .iter()
        .map(|&(z, f)| {
            let en = electronegativity(z);
            (en - mean_en).powi(2) * f
        })
        .sum();
    features[2] = en_var.sqrt();

    // ── 3: Weighted mean atomic radius ──
    let mean_r: f64 = composition
        .iter()
        .map(|&(z, f)| covalent_radius(z) * f)
        .sum();
    features[3] = mean_r;

    // ── 4: Atomic radius range ──
    let radii: Vec<f64> = composition
        .iter()
        .map(|&(z, _)| covalent_radius(z))
        .collect();
    let r_max = radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let r_min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
    features[4] = r_max - r_min;

    // ── 5: Weighted mean ionization energy (eV) ──
    let mean_ie: f64 = composition
        .iter()
        .map(|&(z, f)| first_ionization_energy(z) * f)
        .sum();
    features[5] = mean_ie;

    // ── 6: Weighted mean electron affinity (eV) ──
    let mean_ea: f64 = composition
        .iter()
        .map(|&(z, f)| electron_affinity(z) * f)
        .sum();
    features[6] = mean_ea;

    // ── 7: Weighted mean atomic mass (amu) ──
    let mean_mass: f64 = composition.iter().map(|&(z, f)| atomic_mass(z) * f).sum();
    features[7] = mean_mass;

    // ── 8: Total valence electrons per atom ──
    let total_val: f64 = composition
        .iter()
        .map(|&(z, f)| valence_electrons(z) as f64 * f)
        .sum();
    features[8] = total_val;

    // ── 9: Number of distinct elements ──
    features[9] = composition.len() as f64;

    // ── 10: Weighted mean period ──
    let mean_period: f64 = composition
        .iter()
        .map(|&(z, f)| element_period(z) as f64 * f)
        .sum();
    features[10] = mean_period;

    // ── 11: Weighted mean group (0 for d/f-block) ──
    let mean_group: f64 = composition
        .iter()
        .map(|&(z, f)| element_group(z) as f64 * f)
        .sum();
    features[11] = mean_group;

    // ── 12: Fraction of d-block elements ──
    let d_frac: f64 = composition
        .iter()
        .filter(|&&(z, _)| element_block(z) == 2)
        .map(|&(_, f)| f)
        .sum();
    features[12] = d_frac;

    // ── 13: Fraction of p-block elements ──
    let p_frac: f64 = composition
        .iter()
        .filter(|&&(z, _)| element_block(z) == 1)
        .map(|&(_, f)| f)
        .sum();
    features[13] = p_frac;

    // ── 14: Max oxidation state estimate ──
    let max_ox: f64 = composition
        .iter()
        .map(|&(z, _)| max_oxidation_state(z) as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    features[14] = max_ox;

    // ── 15: Electronegativity model baseline bandgap ──
    features[15] = electronegativity_bandgap(composition);

    // ── 16: Crystal system ordinal ──
    features[16] = crystal.ordinal() as f64;

    // ── 17: Pettifor chemical scale mean ──
    let mean_pettifor: f64 = composition
        .iter()
        .map(|&(z, f)| pettifor_number(z) * f)
        .sum();
    features[17] = mean_pettifor;

    features
}

// ─── Property lookup functions ───────────────────────────────────────────────

/// Covalent radius in Angstroms.
fn covalent_radius(z: u8) -> f64 {
    if (z as usize) < COVALENT_RADIUS.len() {
        COVALENT_RADIUS[z as usize]
    } else {
        1.5
    }
}

/// First ionization energy in eV (NIST values, Z=1..86).
/// Returns a reasonable estimate for elements beyond Z=86.
fn first_ionization_energy(z: u8) -> f64 {
    // Selected values from NIST (eV). Index = Z. 0 = placeholder.
    const IE: [f64; 87] = [
        0.0,   // 0
        13.60, // H
        24.59, // He
        5.39,  // Li
        9.32,  // Be
        8.30,  // B
        11.26, // C
        14.53, // N
        13.62, // O
        17.42, // F
        21.56, // Ne
        5.14,  // Na
        7.65,  // Mg
        5.99,  // Al
        8.15,  // Si
        10.49, // P
        10.36, // S
        12.97, // Cl
        15.76, // Ar
        4.34,  // K
        6.11,  // Ca
        6.56,  // Sc
        6.83,  // Ti
        6.75,  // V
        6.77,  // Cr
        7.43,  // Mn
        7.90,  // Fe
        7.88,  // Co
        7.64,  // Ni
        7.73,  // Cu
        9.39,  // Zn
        6.00,  // Ga
        7.90,  // Ge
        9.79,  // As
        9.75,  // Se
        11.81, // Br
        14.00, // Kr
        4.18,  // Rb
        5.69,  // Sr
        6.22,  // Y
        6.63,  // Zr
        6.76,  // Nb
        7.09,  // Mo
        7.28,  // Tc
        7.36,  // Ru
        7.46,  // Rh
        8.34,  // Pd
        7.58,  // Ag
        8.99,  // Cd
        5.79,  // In
        7.34,  // Sn
        8.61,  // Sb
        9.01,  // Te
        10.45, // I
        12.13, // Xe
        3.89,  // Cs
        5.21,  // Ba
        5.58,  // La
        5.54,  // Ce
        5.47,  // Pr
        5.53,  // Nd
        5.58,  // Pm
        5.64,  // Sm
        5.67,  // Eu
        6.15,  // Gd
        5.86,  // Tb
        5.94,  // Dy
        6.02,  // Ho
        6.11,  // Er
        6.18,  // Tm
        6.25,  // Yb
        5.43,  // Lu
        6.83,  // Hf
        7.55,  // Ta
        7.86,  // W
        7.83,  // Re
        8.44,  // Os
        8.97,  // Ir
        8.96,  // Pt
        9.23,  // Au
        10.44, // Hg
        6.11,  // Tl
        7.42,  // Pb
        7.29,  // Bi
        8.41,  // Po
        9.30,  // At (estimated)
        10.75, // Rn
    ];
    if (z as usize) < IE.len() {
        IE[z as usize]
    } else {
        6.0 // generic estimate for actinides/superheavy
    }
}

/// Electron affinity in eV (NIST values, Z=1..86).
/// Returns 0.0 for noble gases and elements with negative/unknown EA.
fn electron_affinity(z: u8) -> f64 {
    const EA: [f64; 87] = [
        0.0,  // 0
        0.75, // H
        0.00, // He
        0.62, // Li
        0.00, // Be (negative)
        0.28, // B
        1.26, // C
        0.00, // N (negative)
        1.46, // O
        3.40, // F
        0.00, // Ne
        0.55, // Na
        0.00, // Mg
        0.43, // Al
        1.39, // Si
        0.75, // P
        2.08, // S
        3.61, // Cl
        0.00, // Ar
        0.50, // K
        0.02, // Ca
        0.19, // Sc
        0.08, // Ti
        0.53, // V
        0.68, // Cr
        0.00, // Mn
        0.15, // Fe
        0.66, // Co
        1.16, // Ni
        1.24, // Cu
        0.00, // Zn
        0.43, // Ga
        1.23, // Ge
        0.81, // As
        2.02, // Se
        3.36, // Br
        0.00, // Kr
        0.49, // Rb
        0.05, // Sr
        0.31, // Y
        0.43, // Zr
        0.92, // Nb
        0.75, // Mo
        0.55, // Tc
        1.05, // Ru
        1.14, // Rh
        0.56, // Pd
        1.30, // Ag
        0.00, // Cd
        0.38, // In
        1.11, // Sn
        1.05, // Sb
        1.97, // Te
        3.06, // I
        0.00, // Xe
        0.47, // Cs
        0.14, // Ba
        0.47, // La
        0.65, // Ce
        0.96, // Pr (estimated)
        0.96, // Nd (estimated)
        0.13, // Pm (estimated)
        0.16, // Sm (estimated)
        0.86, // Eu
        0.14, // Gd (estimated)
        1.17, // Tb (estimated)
        0.35, // Dy (estimated)
        0.34, // Ho (estimated)
        0.31, // Er (estimated)
        1.03, // Tm
        0.00, // Yb
        0.34, // Lu
        0.02, // Hf
        0.32, // Ta
        0.82, // W
        0.15, // Re
        1.10, // Os
        1.56, // Ir
        2.13, // Pt
        2.31, // Au
        0.00, // Hg
        0.38, // Tl
        0.36, // Pb
        0.94, // Bi
        1.90, // Po
        2.80, // At (estimated)
        0.00, // Rn
    ];
    if (z as usize) < EA.len() {
        EA[z as usize]
    } else {
        0.5
    }
}

/// Standard atomic mass in amu.
fn atomic_mass(z: u8) -> f64 {
    const MASS: [f64; 87] = [
        0.0,   // 0
        1.008, // H
        4.003, // He
        6.941, // Li
        9.012, // Be
        10.81, // B
        12.01, // C
        14.01, // N
        16.00, // O
        19.00, // F
        20.18, // Ne
        22.99, // Na
        24.31, // Mg
        26.98, // Al
        28.09, // Si
        30.97, // P
        32.07, // S
        35.45, // Cl
        39.95, // Ar
        39.10, // K
        40.08, // Ca
        44.96, // Sc
        47.87, // Ti
        50.94, // V
        52.00, // Cr
        54.94, // Mn
        55.85, // Fe
        58.93, // Co
        58.69, // Ni
        63.55, // Cu
        65.38, // Zn
        69.72, // Ga
        72.63, // Ge
        74.92, // As
        78.97, // Se
        79.90, // Br
        83.80, // Kr
        85.47, // Rb
        87.62, // Sr
        88.91, // Y
        91.22, // Zr
        92.91, // Nb
        95.95, // Mo
        97.00, // Tc
        101.1, // Ru
        102.9, // Rh
        106.4, // Pd
        107.9, // Ag
        112.4, // Cd
        114.8, // In
        118.7, // Sn
        121.8, // Sb
        127.6, // Te
        126.9, // I
        131.3, // Xe
        132.9, // Cs
        137.3, // Ba
        138.9, // La
        140.1, // Ce
        140.9, // Pr
        144.2, // Nd
        145.0, // Pm
        150.4, // Sm
        152.0, // Eu
        157.3, // Gd
        158.9, // Tb
        162.5, // Dy
        164.9, // Ho
        167.3, // Er
        168.9, // Tm
        173.0, // Yb
        175.0, // Lu
        178.5, // Hf
        180.9, // Ta
        183.8, // W
        186.2, // Re
        190.2, // Os
        192.2, // Ir
        195.1, // Pt
        197.0, // Au
        200.6, // Hg
        204.4, // Tl
        207.2, // Pb
        209.0, // Bi
        209.0, // Po
        210.0, // At
        222.0, // Rn
    ];
    if (z as usize) < MASS.len() {
        MASS[z as usize]
    } else {
        z as f64 * 2.5 // rough estimate for superheavy
    }
}

/// Typical number of valence electrons for main group / transition metals.
fn valence_electrons(z: u8) -> u8 {
    match z {
        1 => 1,
        2 => 2, // H, He
        3 => 1,
        4 => 2, // Li, Be
        5 => 3,
        6 => 4,
        7 => 5,
        8 => 6,
        9 => 7,
        10 => 8, // B-Ne
        11 => 1,
        12 => 2, // Na, Mg
        13 => 3,
        14 => 4,
        15 => 5,
        16 => 6,
        17 => 7,
        18 => 8, // Al-Ar
        19 => 1,
        20 => 2, // K, Ca
        21 => 3,
        22 => 4,
        23 => 5,
        24 => 6,
        25 => 7,
        26 => 8,
        27 => 9,
        28 => 10,
        29 => 11,
        30 => 2, // Sc-Zn
        31 => 3,
        32 => 4,
        33 => 5,
        34 => 6,
        35 => 7,
        36 => 8, // Ga-Kr
        37 => 1,
        38 => 2, // Rb, Sr
        39 => 3,
        40 => 4,
        41 => 5,
        42 => 6,
        43 => 7,
        44 => 8,
        45 => 9,
        46 => 10,
        47 => 11,
        48 => 2, // Y-Cd
        49 => 3,
        50 => 4,
        51 => 5,
        52 => 6,
        53 => 7,
        54 => 8, // In-Xe
        55 => 1,
        56 => 2,      // Cs, Ba
        57..=71 => 3, // Lanthanides (approx.)
        72 => 4,
        73 => 5,
        74 => 6,
        75 => 7,
        76 => 8,
        77 => 9,
        78 => 10,
        79 => 11,
        80 => 2, // Hf-Hg
        81 => 3,
        82 => 4,
        83 => 5,
        84 => 6,
        85 => 7,
        86 => 8, // Tl-Rn
        _ => 4,  // Generic fallback
    }
}

/// Maximum common oxidation state for an element.
fn max_oxidation_state(z: u8) -> i8 {
    match z {
        1 => 1,
        2 => 0,
        3 => 1,
        4 => 2,
        5 => 3,
        6 => 4,
        7 => 5,
        8 => -2,
        9 => -1,
        10 => 0,
        11 => 1,
        12 => 2,
        13 => 3,
        14 => 4,
        15 => 5,
        16 => 6,
        17 => 7,
        18 => 0,
        19 => 1,
        20 => 2,
        21 => 3,
        22 => 4,
        23 => 5,
        24 => 6,
        25 => 7,
        26 => 3,
        27 => 3,
        28 => 3,
        29 => 2,
        30 => 2,
        31 => 3,
        32 => 4,
        33 => 5,
        34 => 6,
        35 => 5,
        36 => 2,
        37 => 1,
        38 => 2,
        39 => 3,
        40 => 4,
        41 => 5,
        42 => 6,
        43 => 7,
        44 => 8,
        45 => 4,
        46 => 4,
        47 => 1,
        48 => 2,
        49 => 3,
        50 => 4,
        51 => 5,
        52 => 6,
        53 => 7,
        54 => 8,
        55 => 1,
        56 => 2,
        57 => 3,
        58 => 4,
        59 => 4,
        60 => 3,
        64 => 3,
        72 => 4,
        73 => 5,
        74 => 6,
        75 => 7,
        80 => 2,
        81 => 3,
        82 => 4,
        83 => 5,
        _ => 3, // Generic fallback
    }
}

/// Pettifor chemical scale number (Pettifor, 1984).
///
/// Orders elements by chemical similarity rather than atomic number.
/// Values range from ~0.0 (electropositive metals) to ~100.0 (noble gases).
/// Used to capture chemical bonding trends that Z alone misses.
fn pettifor_number(z: u8) -> f64 {
    // Mendeleev numbers / Pettifor scale (approximate, 103 elements).
    // Based on Pettifor, D.G. (1984). A chemical scale for crystal-structure maps.
    // Solid State Communications 51, 31-34.
    // Normalized to 0-102 range. Index = Z.
    const PETTIFOR: [f64; 104] = [
        0.0,  // 0: placeholder
        92.0, // 1: H
        98.0, // 2: He
        1.0,  // 3: Li
        67.0, // 4: Be
        72.0, // 5: B
        77.0, // 6: C
        82.0, // 7: N
        87.0, // 8: O
        93.0, // 9: F
        97.0, // 10: Ne
        2.0,  // 11: Na
        66.0, // 12: Mg
        68.0, // 13: Al
        73.0, // 14: Si
        78.0, // 15: P
        83.0, // 16: S
        88.0, // 17: Cl
        96.0, // 18: Ar
        3.0,  // 19: K
        7.0,  // 20: Ca
        14.0, // 21: Sc
        44.0, // 22: Ti
        46.0, // 23: V
        48.0, // 24: Cr
        50.0, // 25: Mn
        52.0, // 26: Fe
        54.0, // 27: Co
        56.0, // 28: Ni
        58.0, // 29: Cu
        64.0, // 30: Zn
        69.0, // 31: Ga
        74.0, // 32: Ge
        79.0, // 33: As
        84.0, // 34: Se
        89.0, // 35: Br
        95.0, // 36: Kr
        4.0,  // 37: Rb
        8.0,  // 38: Sr
        15.0, // 39: Y
        43.0, // 40: Zr
        45.0, // 41: Nb
        47.0, // 42: Mo
        49.0, // 43: Tc
        51.0, // 44: Ru
        53.0, // 45: Rh
        55.0, // 46: Pd
        57.0, // 47: Ag
        63.0, // 48: Cd
        70.0, // 49: In
        75.0, // 50: Sn
        80.0, // 51: Sb
        85.0, // 52: Te
        90.0, // 53: I
        94.0, // 54: Xe
        5.0,  // 55: Cs
        9.0,  // 56: Ba
        16.0, // 57: La
        17.0, // 58: Ce
        18.0, // 59: Pr
        19.0, // 60: Nd
        20.0, // 61: Pm
        21.0, // 62: Sm
        22.0, // 63: Eu
        23.0, // 64: Gd
        24.0, // 65: Tb
        25.0, // 66: Dy
        26.0, // 67: Ho
        27.0, // 68: Er
        28.0, // 69: Tm
        29.0, // 70: Yb
        30.0, // 71: Lu
        42.0, // 72: Hf
        41.0, // 73: Ta
        40.0, // 74: W
        39.0, // 75: Re
        38.0, // 76: Os
        37.0, // 77: Ir
        36.0, // 78: Pt
        35.0, // 79: Au
        62.0, // 80: Hg
        71.0, // 81: Tl
        76.0, // 82: Pb
        81.0, // 83: Bi
        86.0, // 84: Po
        91.0, // 85: At
        99.0, // 86: Rn
        6.0,  // 87: Fr
        10.0, // 88: Ra
        11.0, // 89: Ac
        12.0, // 90: Th
        13.0, // 91: Pa
        31.0, // 92: U
        32.0, // 93: Np
        33.0, // 94: Pu
        34.0, // 95: Am
        34.5, // 96: Cm
        35.0, // 97: Bk
        35.5, // 98: Cf
        36.0, // 99: Es
        36.5, // 100: Fm
        37.0, // 101: Md
        37.5, // 102: No
        38.0, // 103: Lr
    ];
    if (z as usize) < PETTIFOR.len() {
        PETTIFOR[z as usize]
    } else {
        50.0 // midrange fallback
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

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
        // 0: mean EN for Si = 1.90
        assert!((f[0] - 1.90).abs() < 0.01, "Si mean_en = {}", f[0]);
        // 1: EN range for single element = 0
        assert!((f[1]).abs() < 0.01, "Si en_range = {}", f[1]);
        // 9: 1 element
        assert!((f[9] - 1.0).abs() < 0.01);
        // 10: period 3
        assert!((f[10] - 3.0).abs() < 0.01, "Si period = {}", f[10]);
        // 16: crystal = Cubic = 0
        assert!((f[16]).abs() < 0.01);
    }

    #[test]
    fn test_gaas_features() {
        let f = extract_features(&[(31, 0.5), (33, 0.5)], &CrystalSystem::Cubic);
        // EN range: As(2.18) - Ga(1.81) = 0.37
        assert!((f[1] - 0.37).abs() < 0.02, "GaAs en_range = {}", f[1]);
        // 9: 2 elements
        assert!((f[9] - 2.0).abs() < 0.01);
        // 13: p-block fraction should be 1.0 (both Ga and As are p-block)
        assert!((f[13] - 1.0).abs() < 0.01, "GaAs p_frac = {}", f[13]);
        // 17: Pettifor mean = (69 + 79) / 2 = 74
        assert!((f[17] - 74.0).abs() < 0.5, "GaAs pettifor = {}", f[17]);
    }

    #[test]
    fn test_empty() {
        let f = extract_features(&[], &CrystalSystem::Cubic);
        assert!(f.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_nacl_features() {
        let f = extract_features(&[(11, 0.5), (17, 0.5)], &CrystalSystem::Cubic);
        // EN range: Cl(3.16) - Na(0.93) = 2.23
        assert!((f[1] - 2.23).abs() < 0.02, "NaCl en_range = {}", f[1]);
        // 15: baseline bandgap should be positive
        assert!(f[15] > 0.0, "NaCl baseline = {}", f[15]);
    }

    #[test]
    fn test_tio2_d_block_fraction() {
        let f = extract_features(
            &[(22, 1.0 / 3.0), (8, 2.0 / 3.0)],
            &CrystalSystem::Tetragonal,
        );
        // Ti is d-block (fraction = 1/3), O is p-block (fraction = 2/3)
        assert!((f[12] - 1.0 / 3.0).abs() < 0.01, "TiO2 d_frac = {}", f[12]);
        assert!((f[13] - 2.0 / 3.0).abs() < 0.01, "TiO2 p_frac = {}", f[13]);
        // crystal = Tetragonal = 2
        assert!((f[16] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_crystal_system_encoding() {
        let cubic = extract_features(&[(14, 1.0)], &CrystalSystem::Cubic);
        let hex = extract_features(&[(14, 1.0)], &CrystalSystem::Hexagonal);
        let tri = extract_features(&[(14, 1.0)], &CrystalSystem::Triclinic);
        assert!((cubic[16] - 0.0).abs() < 0.01);
        assert!((hex[16] - 1.0).abs() < 0.01);
        assert!((tri[16] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_pettifor_ordering() {
        // Alkali metals should have low Pettifor numbers, halogens high
        assert!(pettifor_number(3) < pettifor_number(9)); // Li < F
        assert!(pettifor_number(11) < pettifor_number(17)); // Na < Cl
        // Transition metals in the middle
        assert!(pettifor_number(26) > pettifor_number(11)); // Fe > Na
        assert!(pettifor_number(26) < pettifor_number(9)); // Fe < F
    }

    #[test]
    fn test_all_features_finite() {
        use crate::training_data::load_training_data;
        let data = load_training_data();
        for entry in &data {
            let f = extract_features(&entry.composition, &entry.crystal_system);
            for (i, &val) in f.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "{}: feature[{}] = {} is not finite",
                    entry.formula,
                    i,
                    val
                );
            }
        }
    }
}
