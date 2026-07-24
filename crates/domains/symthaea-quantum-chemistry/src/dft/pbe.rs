// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PBE (Perdew-Burke-Ernzerhof, 1996) exchange functional -- Phase Q5d,
//! 2026-07-17.
//!
//! ## Status: exchange only, post-hoc evaluation only
//!
//! This module implements the PBE **exchange** enhancement factor and
//! energy density only -- PBE **correlation** (the `H(rs,t,ζ)` function) is
//! a separate, materially more complex implicit-equation formula, not
//! implemented here. `xc::pbe_exchange_energy_posthoc` evaluates this
//! functional **non-self-consistently** on an already-converged density
//! (e.g. from `kohn_sham_dft`'s LDA result) -- it is not wired into the
//! Kohn-Sham SCF/Fock-matrix build, which for a genuine GGA needs an
//! additional gradient-coupling term
//! (`V_μν^GGA = ∫[∂f/∂ρ·φ_μφ_ν + 2·∂f/∂σ·∇ρ·(φ_μ∇φ_ν+φ_ν∇φ_μ)] dr`,
//! Johnson/Gill/Pople 1993) not built here. Do not describe this as a
//! self-consistent PBE calculation.
//!
//! ## Constants: fetched, not memorized
//!
//! `KAPPA = 0.804` and `MU = 0.2195149727645171` fetched directly from
//! libxc's `src/gga_x_pbe.c` (`pbe_values = {0.8040, MU_PBE}`) and
//! `src/util.h` (`MU_PBE = 0.2195149727645171 /* mu = beta*pi^2/3, beta =
//! 0.06672455060314922 */`) via `curl` from `gitlab.com/libxc/libxc`
//! (libxc is hosted on GitLab, not GitHub) -- the standard, canonical
//! open-source XC functional library, not recalled from memory.
//!
//! The reduced-gradient formula (`reduced_gradient`) was independently
//! cross-checked during planning: numerically evaluating libxc's own
//! internal formula chain (`gga_x_pbe.c`'s `xchan`/`pbe_f`/`pbe_f0`, which
//! works in per-spin-channel variables) for a representative `(ρ,|∇ρ|)`
//! pair reproduces the textbook closed form used here to 1 part in 10^15.
//!
//! References:
//! - Perdew, Burke & Ernzerhof (1996). Phys. Rev. Lett. 77, 3865.
//! - libxc: `gitlab.com/libxc/libxc`, `src/gga_x_pbe.c` +
//!   `src/maple2c/gga_exc/gga_x_pbe.c`.

use crate::dft::lda::SlaterExchange;
use std::f64::consts::PI;

/// PBE exchange functional.
pub struct PbeExchange;

impl PbeExchange {
    /// Asymptotic value of the enhancement function (fetched from libxc,
    /// see module doc).
    pub const KAPPA: f64 = 0.804;
    /// Coefficient of the 2nd-order gradient expansion, `= β·π²/3` with
    /// `β = 0.06672455060314922` (fetched from libxc, see module doc).
    pub const MU: f64 = 0.2195149727645171;

    /// Reduced density gradient `s = |∇ρ| / (2·(3π²)^(1/3)·ρ^(4/3))`.
    /// Verified during planning to match libxc's internal formula chain
    /// exactly (see module doc).
    pub fn reduced_gradient(rho: f64, grad_rho_norm: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }
        grad_rho_norm / (2.0 * (3.0 * PI * PI).powf(1.0 / 3.0) * rho.powf(4.0 / 3.0))
    }

    /// PBE exchange enhancement factor `F_x(s) = 1 + κ - κ/(1 + μs²/κ)`.
    /// `F_x(0) = 1` exactly (the `s→0` limit is forced by this formula's
    /// own algebraic structure, not approximate), and `F_x(s) ≥ 1` for all
    /// `s ≥ 0` since `κμs²/(κ+μs²) ≥ 0` whenever `κ,μ > 0` -- both are
    /// checked directly in tests.
    pub fn enhancement_factor(s: f64) -> f64 {
        let kappa = Self::KAPPA;
        let mu = Self::MU;
        1.0 + kappa - kappa / (1.0 + mu * s * s / kappa)
    }

    /// PBE exchange energy density: `ε_x^PBE(ρ,σ) = ε_x^LDA(ρ)·F_x(s)`,
    /// where `σ = |∇ρ|²`. Reduces exactly to `SlaterExchange::energy_density`
    /// when `σ=0` (checked in tests).
    pub fn energy_density(rho: f64, grad_rho_sq: f64) -> f64 {
        if rho < 1e-20 {
            return 0.0;
        }
        let s = Self::reduced_gradient(rho, grad_rho_sq.sqrt());
        SlaterExchange::energy_density(rho) * Self::enhancement_factor(s)
    }

    /// Total PBE exchange energy contribution per grid point: `ε_x^PBE × ρ`.
    pub fn energy_per_point(rho: f64, grad_rho_sq: f64) -> f64 {
        Self::energy_density(rho, grad_rho_sq) * rho
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbe_constants_match_libxc() {
        // Phase Q5d: catches an accidental future edit to these
        // fetched-not-memorized constants.
        assert_eq!(PbeExchange::KAPPA, 0.804);
        assert_eq!(PbeExchange::MU, 0.2195149727645171);
    }

    #[test]
    fn test_enhancement_factor_exact_one_at_zero_gradient() {
        // F_x(0) = 1 + kappa - kappa/(1+0) = 1 exactly -- forced by the
        // formula's own algebra, not approximate.
        let f0 = PbeExchange::enhancement_factor(0.0);
        assert_eq!(f0, 1.0);
    }

    #[test]
    fn test_enhancement_factor_at_least_one() {
        for s in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let f = PbeExchange::enhancement_factor(s);
            assert!(f >= 1.0, "F_x({s}) = {f}, expected >= 1");
        }
    }

    #[test]
    fn test_pbe_exchange_reduces_exactly_to_slater_at_zero_gradient() {
        // The strongest identity: sigma=0 must reproduce the existing,
        // already-tested Slater exchange exactly.
        for rho in [0.01, 0.1, 0.5, 1.0, 2.0] {
            let pbe = PbeExchange::energy_density(rho, 0.0);
            let lda = SlaterExchange::energy_density(rho);
            assert_eq!(pbe, lda, "rho={rho}: PBE(sigma=0)={pbe} != LDA={lda}");
        }
    }

    #[test]
    fn test_pbe_exchange_magnitude_exceeds_lda_for_nonzero_gradient() {
        // |E_x^PBE| >= |E_x^LDA| always, since F_x(s) >= 1 always.
        let rho = 0.15;
        let grad_rho_sq = 0.08 * 0.08;
        let pbe = PbeExchange::energy_density(rho, grad_rho_sq);
        let lda = SlaterExchange::energy_density(rho);
        assert!(
            pbe.abs() >= lda.abs(),
            "|PBE|={} should be >= |LDA|={}",
            pbe.abs(),
            lda.abs()
        );
        assert!(
            pbe < 0.0,
            "PBE exchange energy density should be negative: {pbe}"
        );
    }

    #[test]
    fn test_reduced_gradient_zero_density() {
        assert_eq!(PbeExchange::reduced_gradient(0.0, 1.0), 0.0);
    }
}
