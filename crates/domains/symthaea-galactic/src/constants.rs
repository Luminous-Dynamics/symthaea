// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Physical constants and unit conversions for rotation-curve modeling.
//!
//! All model math runs in SI internally; SPARC data arrives in kpc and km/s.
//! Every conversion lives here so unit errors have exactly one place to hide.

/// Speed of light [m/s]
pub const C_M_S: f64 = 2.997_924_58e8;

/// Newton's constant [m³ kg⁻¹ s⁻²]
pub const G_SI: f64 = 6.674_30e-11;

/// One kiloparsec [m]
pub const KPC_M: f64 = 3.085_677_581_491_367e19;

/// km/s → m/s
pub const KMS_MS: f64 = 1.0e3;

/// Hubble constant assumed by SPARC distance work [km/s/Mpc]
pub const H0_KMS_MPC: f64 = 73.0;

/// Solar mass [kg]
pub const MSUN_KG: f64 = 1.988_92e30;

// ── Mass-to-light ratios at [3.6] (fixed across ALL models for fairness) ────

/// Stellar disk mass-to-light ratio at [3.6] [M☉/L☉] (Lelli+2016 fiducial)
pub const UPSILON_DISK: f64 = 0.5;

/// Bulge mass-to-light ratio at [3.6] [M☉/L☉] (Lelli+2016 fiducial)
pub const UPSILON_BULGE: f64 = 0.7;

/// Helium/metals correction factor on HI mass for total gas mass
pub const GAS_HELIUM_FACTOR: f64 = 1.4;

// ── MOND ─────────────────────────────────────────────────────────────────────

/// MOND acceleration scale a₀ [m/s²] (McGaugh, Lelli & Schombert 2016)
pub const MOND_A0: f64 = 1.2e-10;

// ── Conformal gravity (Mannheim & O'Brien 2012, PRD 85, 124020) ─────────────
// v²(r) = v_bar²(r) + γ★N★c²r/2 + γ₀c²r/2 − κc²r²
// γ★ is per solar mass of baryonic matter (N★ = baryonic mass in M☉);
// γ₀ and κ are universal cosmological terms.

/// Per-solar-mass linear potential coefficient γ★ [1/m] (5.42e-41 cm⁻¹)
pub const CG_GAMMA_STAR: f64 = 5.42e-39;

/// Universal linear potential coefficient γ₀ [1/m] (3.06e-30 cm⁻¹)
pub const CG_GAMMA_0: f64 = 3.06e-28;

/// Universal quadratic cutoff κ [1/m²] (9.54e-54 cm⁻²)
pub const CG_KAPPA: f64 = 9.54e-50;

// ── Numerical guards ─────────────────────────────────────────────────────────

/// Floor on velocity uncertainty [km/s] — purely a division-by-zero guard,
/// NOT a statistical error floor (none is applied; documented in README).
pub const V_ERR_FLOOR_KMS: f64 = 1.0e-3;

/// Convert a squared velocity in (km/s)² at radius r [kpc] into a
/// gravitational acceleration [m/s²]: g = v²/r.
pub fn accel_si(v_sq_kms2: f64, r_kpc: f64) -> f64 {
    v_sq_kms2 * KMS_MS * KMS_MS / (r_kpc * KPC_M)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cgs_to_si_conversions_are_consistent() {
        // 1 cm⁻¹ = 100 m⁻¹; original values are quoted in cm⁻¹/cm⁻²
        assert!((CG_GAMMA_STAR / 100.0 - 5.42e-41).abs() < 1e-50);
        assert!((CG_GAMMA_0 / 100.0 - 3.06e-30).abs() < 1e-39);
        assert!((CG_KAPPA / 1.0e4 - 9.54e-54).abs() < 1e-63);
    }

    #[test]
    fn accel_scale_sanity() {
        // A 150 km/s curve at 10 kpc sits near the MOND scale a0 —
        // this is the empirical heart of the rotation-curve problem.
        let g = accel_si(150.0 * 150.0, 10.0);
        assert!(g > 0.1 * MOND_A0 && g < 10.0 * MOND_A0, "g = {g}");
    }

    #[test]
    fn conformal_terms_have_plausible_magnitudes() {
        // At r = 10 kpc with N★ = 1e11 M☉, each conformal term should
        // contribute tens of km/s — not micro- or mega- (unit-error tripwire).
        let r_m = 10.0 * KPC_M;
        let n_star = 1.0e11;
        let v_sq_star = CG_GAMMA_STAR * n_star * C_M_S * C_M_S * r_m / 2.0;
        let v_sq_0 = CG_GAMMA_0 * C_M_S * C_M_S * r_m / 2.0;
        let v_star_kms = v_sq_star.sqrt() / KMS_MS;
        let v_0_kms = v_sq_0.sqrt() / KMS_MS;
        assert!(
            (10.0..300.0).contains(&v_star_kms),
            "γ★ term → {v_star_kms} km/s"
        );
        assert!((10.0..300.0).contains(&v_0_kms), "γ₀ term → {v_0_kms} km/s");
        // κ correction stays subdominant at 10 kpc
        let v_sq_kappa = CG_KAPPA * C_M_S * C_M_S * r_m * r_m;
        assert!(v_sq_kappa < v_sq_star + v_sq_0);
    }
}
