//! # Core Physics: Gamow Integration and Q Factor
//!
//! Rigorous D-D fusion physics with honest Q factor assessment.

use crate::constants::*;
use serde::{Deserialize, Serialize};

/// Result of Gamow peak thermal integration for D-D reaction rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamowResult {
    /// Thermally averaged reaction rate <σv> (cm³/s)
    pub sigma_v_cm3_s: f64,
    /// Gamow peak energy (keV)
    pub gamow_peak_kev: f64,
    /// Gamow peak width (keV)
    pub gamow_width_kev: f64,
    /// Screening enhancement factor
    pub screening_enhancement: f64,
    /// Phonon enhancement factor
    pub phonon_enhancement: f64,
    /// Lattice temperature used (K)
    pub temperature_k: f64,
    /// Screening energy used (eV)
    pub screening_ev: f64,
    /// Number of phonon modes
    pub phonon_modes: u32,
}

impl GamowResult {
    /// Convert reaction rate to neutron production rate.
    ///
    /// `n_d_cm3`: Deuterium number density (atoms/cm³)
    /// `volume_cm3`: Active volume (cm³)
    pub fn to_neutron_rate(&self, n_d_cm3: f64, volume_cm3: f64) -> f64 {
        // Reaction rate density: R = n² × <σv> / 4 (factor of 4 for identical particles)
        let reaction_rate_density = n_d_cm3.powi(2) * self.sigma_v_cm3_s / 4.0;

        // Total reactions per second
        let total_rate = reaction_rate_density * volume_cm3;

        // ~50% go to neutron channel
        total_rate * 0.5
    }

    /// Compute fusion power density (W/cm³).
    pub fn fusion_power_density(&self, n_d_cm3: f64) -> f64 {
        let reaction_rate_density = n_d_cm3.powi(2) * self.sigma_v_cm3_s / 4.0;
        let e_fusion_j = DD_Q_AVERAGE * 1.602e-13; // MeV to J
        reaction_rate_density * e_fusion_j
    }
}

/// Gamow integration calculator.
pub struct GamowIntegration;

impl GamowIntegration {
    /// Full Gamow peak thermal integration for D-D.
    ///
    /// Uses 200-point trapezoidal integration in log-space to handle
    /// the astronomically small values at room temperature.
    ///
    /// # Arguments
    /// * `t_k` - Lattice temperature (K)
    /// * `ue_ev` - Screening energy (eV), use 309 for Pd
    /// * `phonon_modes` - Number of coherent phonon modes (0-5)
    pub fn dd_rate(t_k: f64, ue_ev: f64, phonon_modes: u32) -> GamowResult {
        let kt_kev = KB_KEV * t_k;

        if kt_kev <= 0.0 {
            return GamowResult {
                sigma_v_cm3_s: 0.0,
                gamow_peak_kev: 0.0,
                gamow_width_kev: 0.0,
                screening_enhancement: 1.0,
                phonon_enhancement: 1.0,
                temperature_k: t_k,
                screening_ev: ue_ev,
                phonon_modes,
            };
        }

        // Gamow peak: E_0 = (B_G² × kT² / 4)^(1/3)
        let b_g = DD_GAMOW_CONSTANT;
        let e0_kev = (b_g * b_g * kt_kev * kt_kev / 4.0).powf(1.0 / 3.0);

        // Gamow width: Δ = 4√(E_0×kT/3)
        let delta_kev = 4.0 * (e0_kev * kt_kev / 3.0).sqrt();

        // Phonon boost
        let phonon_boost_kev = (phonon_modes as f64) * PDD_PHONON_KEV;

        // Temperature-dependent screening
        let ue_kev = Self::screening_at_temperature(ue_ev, t_k) / 1000.0;

        // Integration range: ±2.5 Gamow widths
        let e_min = (e0_kev - 2.5 * delta_kev).max(0.001);
        let e_max = e0_kev + 2.5 * delta_kev;
        let n_points: usize = 200;
        let de = (e_max - e_min) / (n_points as f64);

        // 200-point trapezoidal integration in log space
        let mut log_integrand_values: Vec<f64> = Vec::with_capacity(n_points + 1);
        let mut energies: Vec<f64> = Vec::with_capacity(n_points + 1);

        for i in 0..=n_points {
            let e_kev = e_min + (i as f64) * de;
            let e_eff = e_kev + phonon_boost_kev;

            // Log of cross-section with screening
            let e_screened = e_eff + ue_kev;
            let log_sigma = (DD_S_FACTOR / e_screened).ln() - b_g / e_screened.sqrt();

            // Log of integrand
            let log_val = log_sigma + e_kev.ln() - e_kev / kt_kev;

            energies.push(e_kev);
            log_integrand_values.push(log_val);
        }

        // Find maximum for numerical stability
        let log_max = log_integrand_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Trapezoidal integration in shifted space
        let mut integral = 0.0;
        for i in 0..n_points {
            let f0 = (log_integrand_values[i] - log_max).exp();
            let f1 = (log_integrand_values[i + 1] - log_max).exp();
            integral += 0.5 * (f0 + f1) * de;
        }

        let log_integral = log_max + integral.ln();

        // Convert to <σv> in cm³/s
        let mu_amu: f64 = 1.007;
        let prefactor: f64 = 3.727e-12 / (mu_amu.sqrt() * kt_kev.powf(1.5));
        let log_sigma_v = prefactor.ln() + log_integral;
        let sigma_v = log_sigma_v.exp();

        // Compute bare (no screening) for enhancement factor
        let bare_integral = {
            let mut log_bare: Vec<f64> = Vec::with_capacity(n_points + 1);
            for i in 0..=n_points {
                let e_kev = energies[i];
                let log_sigma = (DD_S_FACTOR / e_kev).ln() - b_g / e_kev.sqrt();
                let log_val = log_sigma + e_kev.ln() - e_kev / kt_kev;
                log_bare.push(log_val);
            }
            let log_max_b = log_bare.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut int_b = 0.0;
            for i in 0..n_points {
                let f0 = (log_bare[i] - log_max_b).exp();
                let f1 = (log_bare[i + 1] - log_max_b).exp();
                int_b += 0.5 * (f0 + f1) * de;
            }
            log_max_b + int_b.ln()
        };

        let screening_enhancement = (log_integral - bare_integral).exp();

        // Phonon enhancement
        let phonon_enhancement = if phonon_modes > 0 {
            let no_phonon = Self::dd_rate(t_k, ue_ev, 0);
            if no_phonon.sigma_v_cm3_s > 0.0 {
                sigma_v / no_phonon.sigma_v_cm3_s
            } else {
                1.0
            }
        } else {
            1.0
        };

        GamowResult {
            sigma_v_cm3_s: sigma_v,
            gamow_peak_kev: e0_kev,
            gamow_width_kev: delta_kev,
            screening_enhancement,
            phonon_enhancement,
            temperature_k: t_k,
            screening_ev: ue_ev,
            phonon_modes,
        }
    }

    /// Temperature-dependent screening energy.
    ///
    /// At 300K: returns measured value
    /// Above ~800K: transitions to Debye model (Ue ∝ √T)
    pub fn screening_at_temperature(ue_measured_ev: f64, t_k: f64) -> f64 {
        let ue_room = ue_measured_ev;
        let ue_debye = ue_measured_ev * (t_k / 300.0).sqrt();
        // Gaussian crossover
        let f = (-(t_k - 300.0).powi(2) / (2.0 * 1000.0_f64.powi(2))).exp();
        f * ue_room + (1.0 - f) * ue_debye
    }

    /// Assenbaum screening enhancement at specific energy.
    pub fn screening_enhancement(ue_ev: f64, e_cm_ev: f64) -> f64 {
        let e_kev = e_cm_ev / 1000.0;
        let ue_kev = ue_ev / 1000.0;

        if e_kev <= 0.01 {
            return 1.0;
        }

        let e_screened = e_kev + ue_kev;
        let exponent = DD_GAMOW_CONSTANT * (1.0 / e_kev.sqrt() - 1.0 / e_screened.sqrt());
        let prefactor = e_kev / e_screened;

        (prefactor * exponent.exp()).max(1.0).min(1e6)
    }
}

/// Parameters for Q factor calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QFactorParams {
    /// Deuterium number density (atoms/cm³)
    pub n_d_cm3: f64,
    /// Energy confinement time (seconds)
    pub tau_e_s: f64,
    /// Input power density (W/cm³)
    pub input_power_w_cm3: f64,
}

impl QFactorParams {
    /// Typical LCF conditions in PdD lattice.
    pub fn lcf_typical() -> Self {
        // Pd atoms/cm³ = ρ × N_A / A
        let pd_per_cm3 = PD_DENSITY * 6.022e23 / PD_ATOMIC_MASS;
        let n_d = pd_per_cm3 * 0.7; // D/Pd = 0.7

        Self {
            n_d_cm3: n_d,
            tau_e_s: 1.0,
            input_power_w_cm3: 1.0,
        }
    }

    /// Custom parameters.
    pub fn new(n_d_cm3: f64, tau_e_s: f64, input_power_w_cm3: f64) -> Self {
        Self {
            n_d_cm3,
            tau_e_s,
            input_power_w_cm3,
        }
    }
}

/// Q factor calculator - the honest assessment.
pub struct QFactor;

impl QFactor {
    /// Compute Q = P_fusion / P_input.
    ///
    /// Returns (Q, is_breakeven_achievable).
    ///
    /// **Key result**: At room temperature, Q << 1. This is fundamental physics.
    pub fn compute(gamow: &GamowResult, params: &QFactorParams) -> (f64, bool) {
        let p_fusion = gamow.fusion_power_density(params.n_d_cm3);

        let q = if params.input_power_w_cm3 > 0.0 {
            p_fusion / params.input_power_w_cm3
        } else {
            0.0
        };

        (q, q > 1.0)
    }

    /// Compute temperature required for Q = 1 (if achievable).
    ///
    /// Uses binary search to find T where Q crosses 1.
    pub fn temperature_for_breakeven(params: &QFactorParams) -> Option<f64> {
        let mut t_low = 300.0;
        let mut t_high = 1e9; // 1 billion K

        for _ in 0..100 {
            let t_mid = (t_low + t_high) / 2.0;
            let gamow = GamowIntegration::dd_rate(t_mid, SCREENING_PD_EV, 0);
            let (q, _) = Self::compute(&gamow, params);

            if q > 1.0 {
                t_high = t_mid;
            } else {
                t_low = t_mid;
            }

            if (t_high - t_low) < 1000.0 {
                break;
            }
        }

        let final_gamow = GamowIntegration::dd_rate(t_high, SCREENING_PD_EV, 0);
        let (q, _) = Self::compute(&final_gamow, params);

        if q >= 1.0 {
            Some(t_high)
        } else {
            None // Can't reach breakeven even at extreme temperatures
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamow_at_room_temperature() {
        let result = GamowIntegration::dd_rate(300.0, SCREENING_PD_EV, 2);

        assert!(result.sigma_v_cm3_s > 0.0);
        assert!(result.gamow_peak_kev > 0.0);
        assert!(result.screening_enhancement >= 1.0);
    }

    #[test]
    fn test_q_less_than_one_at_room_temp() {
        let gamow = GamowIntegration::dd_rate(300.0, SCREENING_PD_EV, 2);
        let params = QFactorParams::lcf_typical();
        let (q, achievable) = QFactor::compute(&gamow, &params);

        // This is the critical physics honesty test
        assert!(q < 1.0, "Q must be < 1 at room temperature, got {}", q);
        assert!(!achievable, "Breakeven should NOT be achievable at 300K");
    }

    #[test]
    fn test_q_increases_with_temperature() {
        let params = QFactorParams::lcf_typical();

        let gamow_low = GamowIntegration::dd_rate(1000.0, SCREENING_PD_EV, 0);
        let gamow_high = GamowIntegration::dd_rate(10000.0, SCREENING_PD_EV, 0);

        let (q_low, _) = QFactor::compute(&gamow_low, &params);
        let (q_high, _) = QFactor::compute(&gamow_high, &params);

        assert!(q_high > q_low, "Q should increase with temperature");
    }

    #[test]
    fn test_screening_enhancement() {
        let enhancement = GamowIntegration::screening_enhancement(309.0, 1000.0);
        assert!(enhancement > 1.0, "Screening should enhance at 1 keV");

        let high_e = GamowIntegration::screening_enhancement(309.0, 100000.0);
        assert!(high_e < enhancement, "Enhancement should decrease at high energy");
    }
}
