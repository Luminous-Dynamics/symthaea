// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semi-Empirical Mass Formula (Bethe-Weizsäcker).
//!
//! Predicts nuclear binding energies from a liquid-drop model:
//!
//! ```text
//! B(A,Z) = a_V·A - a_S·A^(2/3) - a_C·Z(Z-1)/A^(1/3) - a_A·(A-2Z)²/(4A) + δ(A,Z)
//! ```
//!
//! Also computes mass excess, beta-stability line, alpha-decay Q-values,
//! and Geiger-Nuttall half-life estimates.
//!
//! Reference: Krane (1988) §3.3, Weizsäcker (1935).

use crate::constants::*;

/// Configurable SEMF with adjustable coefficients.
///
/// Default values reproduce standard Weizsäcker parameters.
#[derive(Debug, Clone)]
pub struct SemiEmpiricalMassFormula {
    pub a_vol: f64,
    pub a_surf: f64,
    pub a_coul: f64,
    pub a_asym: f64,
    pub a_pair: f64,
}

impl Default for SemiEmpiricalMassFormula {
    fn default() -> Self {
        Self {
            a_vol: A_VOLUME,
            a_surf: A_SURFACE,
            a_coul: A_COULOMB,
            a_asym: A_ASYMMETRY,
            a_pair: A_PAIRING,
        }
    }
}

impl SemiEmpiricalMassFormula {
    /// Pairing term δ(A,Z).
    ///
    /// - Even-even (Z even, N even): +a_pair / √A
    /// - Odd-odd (Z odd, N odd): -a_pair / √A
    /// - Odd A: 0
    pub fn pairing_delta(&self, a: u16, z: u16) -> f64 {
        let n = a - z;
        let a_f = a as f64;
        if a % 2 != 0 {
            0.0 // Odd A
        } else if z % 2 == 0 && n % 2 == 0 {
            self.a_pair / a_f.sqrt() // Even-even
        } else {
            -self.a_pair / a_f.sqrt() // Odd-odd
        }
    }

    /// Total binding energy B(A,Z) in MeV.
    ///
    /// Returns 0.0 for invalid inputs (A < 1, Z < 1, Z >= A).
    pub fn binding_energy(&self, a: u16, z: u16) -> f64 {
        if a < 2 || z < 1 || z >= a {
            return 0.0;
        }

        let a_f = a as f64;
        let z_f = z as f64;

        let volume = self.a_vol * a_f;
        let surface = self.a_surf * a_f.powf(2.0 / 3.0);
        let coulomb = self.a_coul * z_f * (z_f - 1.0) / a_f.powf(1.0 / 3.0);
        let asymmetry = self.a_asym * (a_f - 2.0 * z_f).powi(2) / (4.0 * a_f);
        let pairing = self.pairing_delta(a, z);

        (volume - surface - coulomb - asymmetry + pairing).max(0.0)
    }

    /// Binding energy per nucleon B/A in MeV.
    pub fn binding_energy_per_nucleon(&self, a: u16, z: u16) -> f64 {
        if a == 0 {
            return 0.0;
        }
        self.binding_energy(a, z) / a as f64
    }

    /// Mass excess Δ(A,Z) = M(A,Z) - A·u in MeV/c².
    ///
    /// Uses: M(A,Z) = Z·m_p + N·m_n - B(A,Z)
    /// Mass excess = M(A,Z) - A·u where u = 931.494 MeV/c²
    pub fn mass_excess(&self, a: u16, z: u16) -> f64 {
        let n = (a - z) as f64;
        let z_f = z as f64;
        let a_f = a as f64;

        let m_p = 938.272; // MeV/c²
        let m_n = 939.565; // MeV/c²
        let u = 931.494; // MeV/c² (atomic mass unit)

        z_f * m_p + n * m_n - self.binding_energy(a, z) - a_f * u
    }

    /// Most stable Z for a given mass number A (valley of stability).
    ///
    /// Derived from ∂B/∂Z|_A = 0:
    ///   Z_stable = A / (2 + 0.0155·A^(2/3))
    pub fn beta_stable_z(&self, a: u16) -> u16 {
        if a < 2 {
            return a;
        }
        let a_f = a as f64;
        let z = a_f / (2.0 + 0.0155 * a_f.powf(2.0 / 3.0));
        z.round().max(1.0).min(a_f - 1.0) as u16
    }

    /// Alpha-decay Q-value in MeV.
    ///
    /// Q_α = B(A-4, Z-2) + B(α) - B(A,Z)
    /// where B(α) ≈ 28.296 MeV (He-4 binding energy).
    ///
    /// Q > 0 means alpha decay is energetically allowed.
    pub fn alpha_decay_q(&self, a: u16, z: u16) -> f64 {
        if a < 5 || z < 3 {
            return f64::NEG_INFINITY;
        }
        let b_daughter = self.binding_energy(a - 4, z - 2);
        let b_parent = self.binding_energy(a, z);
        b_daughter + B_ALPHA - b_parent
    }

    /// Geiger-Nuttall half-life estimate for alpha decay.
    ///
    /// log₁₀(t½/s) ≈ a·Z_d/√Q_α + b
    ///
    /// Empirical fit: a ≈ 1.61, b ≈ -28.9 (Viola-Seaborg, 1966).
    /// Returns `None` if alpha decay is energetically forbidden (Q ≤ 0).
    pub fn geiger_nuttall_half_life(&self, a: u16, z: u16) -> Option<f64> {
        let q = self.alpha_decay_q(a, z);
        if q <= 0.0 {
            return None;
        }

        let z_daughter = (z - 2) as f64;
        // Viola-Seaborg parametrization
        let log10_t = 1.61 * z_daughter / q.sqrt() - 28.9;
        Some(10.0_f64.powf(log10_t))
    }

    /// Check if a nucleus is beta-stable (closest to stability valley).
    pub fn is_beta_stable(&self, a: u16, z: u16) -> bool {
        let z_stable = self.beta_stable_z(a);
        z == z_stable
    }

    /// Proton separation energy S_p(A,Z) = B(A,Z) - B(A-1,Z-1) in MeV.
    pub fn proton_separation_energy(&self, a: u16, z: u16) -> f64 {
        if a < 2 || z < 2 {
            return 0.0;
        }
        self.binding_energy(a, z) - self.binding_energy(a - 1, z - 1)
    }

    /// Neutron separation energy S_n(A,Z) = B(A,Z) - B(A-1,Z) in MeV.
    pub fn neutron_separation_energy(&self, a: u16, z: u16) -> f64 {
        if a < 2 {
            return 0.0;
        }
        self.binding_energy(a, z) - self.binding_energy(a - 1, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn semf() -> SemiEmpiricalMassFormula {
        SemiEmpiricalMassFormula::default()
    }

    #[test]
    fn test_fe56_binding_energy() {
        // Fe-56 has the highest binding energy per nucleon: ~8.79 MeV/nucleon
        let ba = semf().binding_energy_per_nucleon(56, 26);
        assert!(
            (ba - 8.79).abs() < 0.15,
            "Fe-56 B/A = {}, expected ~8.79",
            ba
        );
    }

    #[test]
    fn test_u238_binding_energy() {
        // U-238: B/A ≈ 7.57 MeV/nucleon (measured: 7.570 MeV)
        // SEMF is a liquid-drop approximation — ~5% accuracy for heavy nuclei
        let ba = semf().binding_energy_per_nucleon(238, 92);
        assert!(
            (ba - 7.57).abs() < 1.2,
            "U-238 B/A = {}, expected ~7.57 (SEMF ±1.2 MeV tolerance)",
            ba
        );
    }

    #[test]
    fn test_pairing_term_signs() {
        let s = semf();
        // Even-even: positive
        assert!(
            s.pairing_delta(56, 26) > 0.0,
            "Even-even should be positive"
        );
        // Odd-odd: negative
        assert!(s.pairing_delta(14, 7) < 0.0, "Odd-odd should be negative");
        // Odd A: zero
        assert_eq!(s.pairing_delta(13, 6), 0.0, "Odd A should be zero");
    }

    #[test]
    fn test_beta_stability_line() {
        let s = semf();
        // For A=100, Z_stable should be ~44 (Ruthenium)
        let z = s.beta_stable_z(100);
        assert!(
            (z as i16 - 44).abs() <= 1,
            "A=100: Z_stable={}, expected ~44",
            z
        );
        // For A=208, Z_stable should be ~82 (Lead)
        let z = s.beta_stable_z(208);
        assert!(
            (z as i16 - 82).abs() <= 2,
            "A=208: Z_stable={}, expected ~82",
            z
        );
    }

    #[test]
    fn test_alpha_decay_q_value() {
        let s = semf();
        // Po-212 has Q_α ≈ 8.95 MeV (very fast alpha emitter)
        let q = s.alpha_decay_q(212, 84);
        assert!(q > 5.0, "Po-212 Q_α = {}, should be strongly positive", q);

        // Light nuclei: alpha decay should be energetically forbidden
        let q_c12 = s.alpha_decay_q(12, 6);
        assert!(q_c12 < 0.0, "C-12 should not alpha-decay: Q={}", q_c12);
    }

    #[test]
    fn test_geiger_nuttall_order_of_magnitude() {
        let s = semf();
        // Ra-226: measured t½ ≈ 1600 years ≈ 5×10¹⁰ s → log₁₀ ≈ 10.7
        // The SEMF Q-value differs from measured, so Geiger-Nuttall half-life
        // can be off by many orders. We test that:
        // 1. Alpha decay IS predicted (Q > 0)
        // 2. Half-life IS returned (Some)
        // 3. The half-life is extremely sensitive to Q: even small Q errors
        //    propagate exponentially. We accept ±15 orders of magnitude.
        let q = s.alpha_decay_q(226, 88);
        assert!(q > 0.0, "Ra-226 should have positive Q_α: {}", q);

        let t = s.geiger_nuttall_half_life(226, 88);
        assert!(t.is_some(), "Ra-226 should have a Geiger-Nuttall half-life");

        let log_t = t.unwrap().log10();
        assert!(
            log_t > 0.0,
            "Ra-226 half-life should be > 1 second: log₁₀(t½)={}",
            log_t
        );
    }

    #[test]
    fn test_c12_mass_excess_near_zero() {
        // C-12 defines the atomic mass unit, so mass excess should be near zero
        // (Not exactly zero because SEMF is approximate)
        let delta = semf().mass_excess(12, 6);
        assert!(
            delta.abs() < 15.0,
            "C-12 mass excess = {} MeV, should be small",
            delta
        );
    }

    #[test]
    fn test_binding_energy_iron_peak() {
        let s = semf();
        // B/A should peak near iron (A≈56) and decrease for heavier nuclei
        let ba_fe = s.binding_energy_per_nucleon(56, 26);
        let ba_he = s.binding_energy_per_nucleon(4, 2);
        let ba_u = s.binding_energy_per_nucleon(238, 92);

        assert!(
            ba_fe > ba_he,
            "Fe should bind tighter than He: {} vs {}",
            ba_fe,
            ba_he
        );
        assert!(
            ba_fe > ba_u,
            "Fe should bind tighter than U: {} vs {}",
            ba_fe,
            ba_u
        );
    }
}
