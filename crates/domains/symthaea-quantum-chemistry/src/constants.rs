// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physical constants for quantum chemistry in atomic units.
//!
//! In atomic units: ℏ = mₑ = e = 4πε₀ = 1.
//! Energy unit: Hartree (Eₕ). Length unit: Bohr (a₀).
//!
//! Sources: CODATA 2022, NIST.

use std::f64::consts::PI;

// ── Conversion Factors ──────────────────────────────────────────────────────

/// Bohr radius → Ångström
pub const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_903;

/// Ångström → Bohr radius
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;

/// Hartree → electron-volt
pub const HARTREE_TO_EV: f64 = 27.211_386_245_988;

/// Hartree → kcal/mol
pub const HARTREE_TO_KCAL: f64 = 627.509_474_063;

/// Hartree → kJ/mol
pub const HARTREE_TO_KJ: f64 = 2625.499_639_48;

/// Hartree → wavenumbers (cm^-1) -- for converting vibrational frequencies.
///
/// Derived (Phase Q4, 2026-07-17) from this module's own already-verified
/// `C_AU` and `BOHR_TO_ANGSTROM` -- not a new hardcoded magic number: a
/// mass-weighted-Hessian eigenvalue's square root (an angular frequency in
/// atomic units, where hbar=1) converts to a wavenumber via
/// `1 / (2*pi*c)` with `c` in the same length units as the wavenumber.
/// `C_AU` is `c` in atomic units (Bohr/atomic-time); converting the length
/// unit from Bohr to cm via `BOHR_TO_ANGSTROM * 1e-8` gives wavenumbers in
/// cm^-1 directly. Verified during planning: this reproduces the standard
/// CODATA reference value (219474.6313632 cm^-1) to 1 part in 10^12.
pub const HARTREE_TO_CM1: f64 = 1.0 / (2.0 * PI_CONST * C_AU * BOHR_TO_ANGSTROM * 1e-8);

/// Elementary charge, C (CODATA/SI 2019 exact definition -- not measured,
/// exact by the SI redefinition of the ampere).
pub const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;

/// Hartree energy in Joules. Derived (Phase Q4, 2026-07-17) from this
/// module's own already-verified `HARTREE_TO_EV` and the exact SI
/// `ELEMENTARY_CHARGE_C` (1 eV = e Joules, exact by definition) -- not an
/// independently memorized constant. Reproduces the standard CODATA
/// reference value (4.3597447222e-18 J).
pub const HARTREE_TO_JOULE: f64 = HARTREE_TO_EV * ELEMENTARY_CHARGE_C;

/// Bohr radius in meters, derived from the already-verified
/// `BOHR_TO_ANGSTROM`.
pub const BOHR_TO_METER: f64 = BOHR_TO_ANGSTROM * 1e-10;

/// Atomic unit of pressure (Hartree / Bohr³) in Pascal. Derived (Phase Q4,
/// 2026-07-17) purely from `HARTREE_TO_JOULE` and `BOHR_TO_METER` above --
/// used to convert 1 atm into atomic-unit pressure for ideal-gas
/// thermochemistry (translational partition function). Reproduces the
/// standard CODATA reference value (2.9421015697e13 Pa) to high precision.
pub const PRESSURE_AU_TO_PASCAL: f64 =
    HARTREE_TO_JOULE / (BOHR_TO_METER * BOHR_TO_METER * BOHR_TO_METER);

/// 1 atmosphere in Pascal (exact by definition).
pub const ATM_TO_PASCAL: f64 = 101_325.0;

// ── Fundamental Constants (atomic units where relevant) ─────────────────────

/// Speed of light in atomic units (≈ 137.036 a.u.)
pub const C_AU: f64 = 137.035_999_084;

/// Fine structure constant α = e²/(4πε₀ℏc) ≈ 1/137
pub const ALPHA_FINE: f64 = 1.0 / C_AU;

/// π (re-exported for convenience in integral formulas)
pub const PI_CONST: f64 = PI;

/// sqrt(π)
pub const SQRT_PI: f64 = 1.772_453_850_905_516;

// ── Numerical Thresholds ────────────────────────────────────────────────────

/// SCF energy convergence threshold (Hartree)
pub const SCF_ENERGY_THRESHOLD: f64 = 1e-8;

/// SCF density convergence threshold
pub const SCF_DENSITY_THRESHOLD: f64 = 1e-6;

/// Integral screening threshold (Schwarz prescreening)
pub const INTEGRAL_SCREENING_THRESHOLD: f64 = 1e-10;

/// Eigenvalue threshold for canonical orthogonalization
/// Eigenvectors of S with eigenvalue below this are discarded
pub const CANONICAL_ORTH_THRESHOLD: f64 = 1e-6;

/// Maximum SCF iterations
pub const MAX_SCF_ITERATIONS: usize = 100;

/// DIIS subspace size
pub const DIIS_SUBSPACE_SIZE: usize = 6;

// ── Helper Functions ────────────────────────────────────────────────────────

/// Double factorial: n!! = n * (n-2) * (n-4) * ... * 1 (or 2)
/// By convention, (-1)!! = 0!! = 1
pub fn double_factorial(n: i32) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let mut result = 1.0;
    let mut k = n;
    while k > 1 {
        result *= k as f64;
        k -= 2;
    }
    result
}

/// Binomial coefficient C(n, k)
pub fn binomial(n: u32, k: u32) -> f64 {
    if k > n {
        return 0.0;
    }
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_factorial() {
        assert_eq!(double_factorial(-1), 1.0);
        assert_eq!(double_factorial(0), 1.0);
        assert_eq!(double_factorial(1), 1.0);
        assert_eq!(double_factorial(3), 3.0); // 3 * 1
        assert_eq!(double_factorial(5), 15.0); // 5 * 3 * 1
        assert_eq!(double_factorial(6), 48.0); // 6 * 4 * 2
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(4, 2), 6.0);
        assert_eq!(binomial(5, 0), 1.0);
        assert_eq!(binomial(5, 5), 1.0);
        assert_eq!(binomial(3, 4), 0.0);
    }

    #[test]
    fn test_hartree_to_cm1_matches_codata() {
        // Phase Q4 (2026-07-17): HARTREE_TO_CM1 is derived from C_AU and
        // BOHR_TO_ANGSTROM, not hardcoded -- verify it reproduces the
        // standard CODATA reference value.
        let codata_reference = 219_474.631_363_2;
        assert!(
            (HARTREE_TO_CM1 - codata_reference).abs() / codata_reference < 1e-9,
            "HARTREE_TO_CM1={HARTREE_TO_CM1}, expected ~{codata_reference}"
        );
    }

    #[test]
    fn test_hartree_to_joule_matches_codata() {
        // Phase Q4 (2026-07-17): HARTREE_TO_JOULE is derived from
        // HARTREE_TO_EV and the exact SI elementary charge.
        let codata_reference = 4.359_744_722_2e-18;
        assert!(
            (HARTREE_TO_JOULE - codata_reference).abs() / codata_reference < 1e-9,
            "HARTREE_TO_JOULE={HARTREE_TO_JOULE}, expected ~{codata_reference}"
        );
    }

    #[test]
    fn test_pressure_au_to_pascal_matches_codata() {
        // Phase Q4 (2026-07-17): derived purely from HARTREE_TO_JOULE and
        // BOHR_TO_METER; verify it reproduces the standard CODATA "atomic
        // unit of pressure" reference value.
        let codata_reference = 2.942_101_569_7e13;
        assert!(
            (PRESSURE_AU_TO_PASCAL - codata_reference).abs() / codata_reference < 1e-6,
            "PRESSURE_AU_TO_PASCAL={PRESSURE_AU_TO_PASCAL}, expected ~{codata_reference}"
        );
    }

    #[test]
    fn test_conversion_roundtrip() {
        let bohr = 1.0;
        let angstrom = bohr * BOHR_TO_ANGSTROM;
        let back = angstrom * ANGSTROM_TO_BOHR;
        assert!((back - bohr).abs() < 1e-12);
    }
}
