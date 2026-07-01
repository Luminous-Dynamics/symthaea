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
    fn test_conversion_roundtrip() {
        let bohr = 1.0;
        let angstrom = bohr * BOHR_TO_ANGSTROM;
        let back = angstrom * ANGSTROM_TO_BOHR;
        assert!((back - bohr).abs() < 1e-12);
    }
}
