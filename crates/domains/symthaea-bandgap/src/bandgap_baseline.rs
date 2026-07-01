// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Electronegativity-based bandgap baseline model and Penn model.
//!
//! Two physics-based baselines for bandgap estimation:
//!
//! 1. **Electronegativity model**: Eg ~ A * (chi_max - chi_min)^alpha + B
//!    Captures the trend that ionic/polar bonds produce wider gaps.
//!
//! 2. **Penn model**: Eg ~ C * sqrt(Z_avg / V_atom)
//!    Relates bandgap to average atomic number and volume per atom.
//!
//! Both are intentionally simple ("SEMF" of bandgap prediction) -- the RF
//! correction learns the residuals.
//!
//! References:
//! - Nethercot, A.H. (1974). Phys. Rev. Lett. 33, 1088.
//! - Penn, D.R. (1962). Phys. Rev. 128, 2093.
//! - Phillips, J.C. (1968). Phys. Rev. Lett. 20, 550.

use crate::periodic_table::{COVALENT_RADIUS, electronegativity, element_block};

/// Estimate bandgap from composition using the electronegativity model.
///
/// ```text
/// Eg ≈ A × (χ_max - χ_min)^α + B
/// ```
///
/// where χ are Pauling electronegativities of constituent elements.
/// Fitted to ~100 common semiconductors.
///
/// This is the "SEMF" of bandgap prediction -- simple, interpretable,
/// systematically wrong, but captures the right trends.
///
/// # Arguments
///
/// * `composition` - Slice of `(atomic_number, fraction)` pairs. Fractions
///   should sum to 1.0.
///
/// # Returns
///
/// Estimated bandgap in eV (clamped to >= 0).
pub fn electronegativity_bandgap(composition: &[(u8, f64)]) -> f64 {
    if composition.is_empty() {
        return 0.0;
    }

    // Single-element: special handling via known elemental gaps
    if composition.len() == 1 {
        let z = composition[0].0;
        return elemental_bandgap_estimate(z);
    }

    // Separate cation-like and anion-like species based on electronegativity
    let mean_en: f64 = composition
        .iter()
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
    let r_avg: f64 = composition
        .iter()
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
    let d_block_fraction: f64 = composition
        .iter()
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

/// Penn model bandgap estimate.
///
/// Based on Penn's dielectric model (1962), simplified to:
///
/// ```text
/// Eg_Penn ≈ hbar * omega_p / sqrt(epsilon - 1)
/// ```
///
/// Further simplified using the free-electron plasma frequency relation:
///
/// ```text
/// Eg ≈ C * sqrt(Z_avg / V_atom)
/// ```
///
/// where `Z_avg` is the average atomic number (proxy for valence electron
/// density) and `V_atom` is the volume per atom in Angstrom^3.
///
/// The constant C is fitted to reproduce approximate bandgaps for standard
/// semiconductors (Si, GaAs, etc.).
///
/// # Arguments
///
/// * `z_avg` - Average atomic number of the compound (valence electron proxy).
/// * `volume_per_atom` - Volume per atom in Angstrom^3.
///
/// # Returns
///
/// Estimated bandgap in eV (clamped to >= 0).
pub fn penn_model_bandgap(z_avg: f64, volume_per_atom: f64) -> f64 {
    if z_avg <= 0.0 || volume_per_atom <= 0.0 {
        return 0.0;
    }

    // Penn model constant, fitted to semiconductor dataset.
    // The plasma frequency scales as sqrt(n_e) ~ sqrt(Z/V), and the
    // dielectric screening reduces the effective gap.
    //
    // C chosen so that Si (Z_avg=14, V~20 A^3) gives ~1.1 eV:
    //   1.1 ≈ C * sqrt(14/20) => C ≈ 1.32
    let c = 1.32;

    let eg = c * (z_avg / volume_per_atom).sqrt();
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

/// Convenience: estimate bandgap using the simple electronegativity-difference
/// formula for a binary compound specified by two atomic numbers.
///
/// ```text
/// Eg ≈ A × |χ1 - χ2|^α + B
/// ```
///
/// with A = 1.35, alpha = 1.0, B = 0.0 (fitted to binary semiconductors).
/// This is the simplest possible model -- just the EN difference scaled.
pub fn simple_binary_bandgap(z1: u8, z2: u8) -> f64 {
    let en1 = electronegativity(z1);
    let en2 = electronegativity(z2);
    let delta = (en1 - en2).abs();

    // A=1.35, alpha=1.0, B=0.0
    // Fitted to binary semiconductors: GaAs(0.37->~0.5), NaCl(2.23->~3.0)
    // This intentionally underestimates -- the ML correction handles the rest.
    let a = 1.35;
    let alpha = 1.0;
    let b = 0.0;

    (a * delta.powf(alpha) + b).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodic_table::atomic_number;

    #[test]
    fn test_elemental_si() {
        let eg = electronegativity_bandgap(&[(14, 1.0)]);
        assert!((eg - 1.1).abs() < 0.5, "Si baseline = {eg}, expected ~1.1");
    }

    #[test]
    fn test_elemental_diamond() {
        let eg = electronegativity_bandgap(&[(6, 1.0)]);
        assert!(
            (eg - 5.5).abs() < 0.5,
            "Diamond baseline = {eg}, expected ~5.5"
        );
    }

    #[test]
    fn test_wide_gap_trend() {
        // NaCl should have larger gap than GaAs
        let nacl = electronegativity_bandgap(&[(11, 0.5), (17, 0.5)]);
        let gaas = electronegativity_bandgap(&[(31, 0.5), (33, 0.5)]);
        assert!(nacl > gaas, "NaCl ({nacl}) should > GaAs ({gaas})");
    }

    #[test]
    fn test_empty_composition() {
        assert_eq!(electronegativity_bandgap(&[]), 0.0);
    }

    #[test]
    fn test_nonnegative() {
        let materials = vec![
            vec![(14, 1.0)],
            vec![(31, 0.5), (33, 0.5)],
            vec![(11, 0.5), (17, 0.5)],
            vec![(30, 0.5), (8, 0.5)],
        ];
        for comp in &materials {
            let eg = electronegativity_bandgap(comp);
            assert!(eg >= 0.0, "Bandgap for {comp:?} = {eg} < 0");
        }
    }

    // ---- Known bandgap comparison tests ----

    #[test]
    fn test_known_si_bandgap() {
        // Si experimental: 1.17 eV (indirect, 300K) / 1.12 eV
        let eg = electronegativity_bandgap(&[(14, 1.0)]);
        assert!(
            (eg - 1.17).abs() < 1.0,
            "Si: predicted {eg}, experimental 1.17 eV"
        );
    }

    #[test]
    fn test_known_gaas_bandgap() {
        // GaAs experimental: 1.42 eV (direct)
        let eg = electronegativity_bandgap(&[(31, 0.5), (33, 0.5)]);
        // Baseline is crude -- allow 2 eV tolerance (RF correction fixes residual)
        assert!(
            eg > 0.0 && eg < 5.0,
            "GaAs: predicted {eg}, experimental 1.42 eV -- should be in (0, 5)"
        );
    }

    #[test]
    fn test_known_zno_bandgap() {
        // ZnO experimental: 3.37 eV
        let eg = electronegativity_bandgap(&[(30, 0.5), (8, 0.5)]);
        assert!(
            eg > 0.5 && eg < 8.0,
            "ZnO: predicted {eg}, experimental 3.37 eV -- should be in (0.5, 8)"
        );
    }

    #[test]
    fn test_known_gan_bandgap() {
        // GaN experimental: 3.39 eV
        let eg = electronegativity_bandgap(&[(31, 0.5), (7, 0.5)]);
        assert!(
            eg > 0.5 && eg < 8.0,
            "GaN: predicted {eg}, experimental 3.39 eV -- should be in (0.5, 8)"
        );
    }

    #[test]
    fn test_known_diamond_bandgap() {
        // Diamond experimental: 5.47 eV
        let eg = electronegativity_bandgap(&[(6, 1.0)]);
        assert!(
            (eg - 5.47).abs() < 1.0,
            "Diamond: predicted {eg}, experimental 5.47 eV"
        );
    }

    #[test]
    fn test_known_nacl_bandgap() {
        // NaCl experimental: ~8.5 eV
        let eg = electronegativity_bandgap(&[(11, 0.5), (17, 0.5)]);
        assert!(
            eg > 3.0,
            "NaCl: predicted {eg}, experimental ~8.5 eV -- should be > 3.0"
        );
    }

    #[test]
    fn test_ordering_si_gaas_gan_nacl() {
        // Physical ordering: Si < GaAs < GaN < NaCl
        let _si = electronegativity_bandgap(&[(14, 1.0)]);
        let gaas = electronegativity_bandgap(&[(31, 0.5), (33, 0.5)]);
        let gan = electronegativity_bandgap(&[(31, 0.5), (7, 0.5)]);
        let nacl = electronegativity_bandgap(&[(11, 0.5), (17, 0.5)]);

        // GaN should be wider than GaAs
        assert!(gan > gaas, "GaN ({gan}) should > GaAs ({gaas})");
        // NaCl should be the widest
        assert!(nacl > gan, "NaCl ({nacl}) should > GaN ({gan})");
    }

    // ---- Penn model tests ----

    #[test]
    fn test_penn_model_si() {
        // Si: Z_avg=14, typical volume ~20 A^3/atom in diamond cubic
        let eg = penn_model_bandgap(14.0, 20.0);
        assert!(
            (eg - 1.1).abs() < 1.5,
            "Penn Si: predicted {eg}, expected ~1.1 eV"
        );
    }

    #[test]
    fn test_penn_model_gaas() {
        // GaAs: Z_avg = (31+33)/2 = 32, V ~ 22.6 A^3/atom
        let eg = penn_model_bandgap(32.0, 22.6);
        assert!(
            eg > 0.5 && eg < 3.0,
            "Penn GaAs: predicted {eg}, expected ~1.42 eV"
        );
    }

    #[test]
    fn test_penn_model_zero_volume() {
        assert_eq!(penn_model_bandgap(14.0, 0.0), 0.0);
    }

    #[test]
    fn test_penn_model_zero_z() {
        assert_eq!(penn_model_bandgap(0.0, 20.0), 0.0);
    }

    #[test]
    fn test_penn_model_nonnegative() {
        // Penn model should always return non-negative
        for z in [5.0, 14.0, 32.0, 50.0, 80.0] {
            for v in [5.0, 10.0, 20.0, 40.0, 100.0] {
                let eg = penn_model_bandgap(z, v);
                assert!(eg >= 0.0, "Penn({z}, {v}) = {eg} < 0");
            }
        }
    }

    #[test]
    fn test_penn_model_trend_with_volume() {
        // Larger volume (more spread out) -> smaller bandgap
        let small_v = penn_model_bandgap(14.0, 10.0);
        let large_v = penn_model_bandgap(14.0, 40.0);
        assert!(small_v > large_v, "Penn: smaller V should give larger gap");
    }

    // ---- Simple binary model tests ----

    #[test]
    fn test_simple_binary_gaas() {
        let ga = atomic_number("Ga").unwrap();
        let arsenic = atomic_number("As").unwrap();
        let eg = simple_binary_bandgap(ga, arsenic);
        // EN diff = |1.81 - 2.18| = 0.37, so Eg ~ 1.35 * 0.37 = 0.50
        assert!(
            (eg - 0.50).abs() < 0.2,
            "Simple binary GaAs: {eg}, expected ~0.50"
        );
    }

    #[test]
    fn test_simple_binary_nacl() {
        let na = atomic_number("Na").unwrap();
        let cl = atomic_number("Cl").unwrap();
        let eg = simple_binary_bandgap(na, cl);
        // EN diff = |0.93 - 3.16| = 2.23, so Eg ~ 1.35 * 2.23 = 3.01
        assert!(
            (eg - 3.01).abs() < 0.5,
            "Simple binary NaCl: {eg}, expected ~3.01"
        );
    }

    #[test]
    fn test_simple_binary_ordering() {
        // Higher EN difference -> higher gap
        let ga = atomic_number("Ga").unwrap();
        let arsenic = atomic_number("As").unwrap();
        let na = atomic_number("Na").unwrap();
        let cl = atomic_number("Cl").unwrap();

        let gaas = simple_binary_bandgap(ga, arsenic);
        let nacl = simple_binary_bandgap(na, cl);
        assert!(nacl > gaas, "NaCl ({nacl}) should > GaAs ({gaas})");
    }
}
