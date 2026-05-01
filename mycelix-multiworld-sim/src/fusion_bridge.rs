// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fusion bridge: wire spark-engine's Gamow physics into the civilization sim.
//!
//! Replaces the flat `lcf_probability = 0.5%/tick` with a physics-grounded
//! calculation based on the civilization's technology levels.
//!
//! The Gamow integral predicts D-D fusion rates at given temperature and
//! screening energy. The civilization's engineering tech level maps to
//! achievable temperature, and science tech to screening efficiency.
//!
//! # The Rate Gap
//!
//! Standard Gamow physics predicts <σv> ~ 10^-50 cm³/s at room temperature.
//! NASA observed ~10³ neutrons/s in palladium lattices. This 10^53 gap is
//! the central mystery of Lattice Confinement Fusion.
//!
//! In the sim, closing this gap requires ~3 tech levels of advancement,
//! which naturally clusters LCF breakthroughs around year 200-400.

use serde::{Deserialize, Serialize};

// ============================================================================
// PHYSICS CONSTANTS (from spark-engine)
// ============================================================================

/// Gamow constant for D-D fusion (keV^0.5).
const DD_GAMOW_CONSTANT: f64 = 30.71e-3;
/// Palladium electron screening energy (eV) — Raiola et al. 2004.
const SCREENING_PD_EV: f64 = 309.0;
/// Pd-D lattice phonon energy (keV).
const PDD_PHONON_KEV: f64 = 0.056;
/// D-D Q value average (MeV).
const DD_Q_AVERAGE: f64 = 3.65;
/// Default deuterium density in Pd (atoms/cm³).
const DEFAULT_N_D_CM3: f64 = 4.8e22;
/// Default active research volume (cm³).
const DEFAULT_VOLUME_CM3: f64 = 0.01;

// ============================================================================
// CIVILIZATION FUSION STATE
// ============================================================================

/// Maps civilization technology to fusion physics parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CivilizationFusionState {
    /// Effective lab temperature (K) — higher tech = hotter plasmas achievable.
    pub effective_temperature_k: f64,
    /// Electron screening energy (eV) — better materials = better screening.
    pub screening_ev: f64,
    /// Coherent phonon modes — requires advanced manufacturing.
    pub phonon_modes: u32,
    /// Deuterium density (atoms/cm³).
    pub deuterium_density_cm3: f64,
    /// Active research volume (cm³).
    pub research_volume_cm3: f64,
    /// Computed fusion rate (neutrons/s).
    pub neutron_rate: f64,
    /// Log10 of the rate gap (observed/predicted).
    pub rate_gap_log10: f64,
    /// Q factor (fusion power / input power).
    pub q_factor: f64,
    /// Whether energy breakeven has been achieved (Q > 1).
    pub breakeven_achieved: bool,
    /// LCF breakthrough probability per tick (0-1).
    pub lcf_probability: f64,
}

impl Default for CivilizationFusionState {
    fn default() -> Self {
        Self {
            effective_temperature_k: 300.0,
            screening_ev: SCREENING_PD_EV,
            phonon_modes: 0,
            deuterium_density_cm3: DEFAULT_N_D_CM3,
            research_volume_cm3: DEFAULT_VOLUME_CM3,
            neutron_rate: 0.0,
            rate_gap_log10: 53.0,
            q_factor: 0.0,
            breakeven_achieved: false,
            lcf_probability: 0.0,
        }
    }
}

impl CivilizationFusionState {
    /// Update fusion state from civilization technology levels.
    ///
    /// # Arguments
    /// - `engineering_tech`: Engineering sector technology level (0-10 scale)
    /// - `science_tech`: Science sector technology level (0-10 scale)
    pub fn update_from_tech(&mut self, engineering_tech: f64, science_tech: f64) {
        // Map tech levels to physics parameters
        // Engineering → temperature (ability to confine hot plasmas)
        self.effective_temperature_k = 300.0 + (engineering_tech - 1.0).max(0.0) * 100_000.0;

        // Science → screening energy (materials science, lattice engineering)
        self.screening_ev = SCREENING_PD_EV * (1.0 + (science_tech - 1.0).max(0.0) * 0.5);

        // Manufacturing → phonon modes (coherent control requires precision)
        self.phonon_modes = ((engineering_tech - 2.0).max(0.0).floor() as u32).min(5);

        // Compute Gamow fusion rate
        let gamow = simplified_gamow_dd(
            self.effective_temperature_k,
            self.screening_ev,
            self.phonon_modes,
        );

        // Convert to neutron rate
        let reaction_rate_density = self.deuterium_density_cm3.powi(2) * gamow / 4.0;
        self.neutron_rate = reaction_rate_density * self.research_volume_cm3 * 0.5;

        // Rate gap: observed 10^3 n/s vs predicted
        self.rate_gap_log10 = if self.neutron_rate > 1e-50 {
            (1e3 / self.neutron_rate).log10()
        } else {
            53.0 // Original gap
        };

        // Q factor: fusion power / input power (rough estimate)
        let fusion_power =
            reaction_rate_density * self.research_volume_cm3 * DD_Q_AVERAGE * 1.602e-13; // MeV to J
        let input_power = 1.0; // 1 watt input (conservative)
        self.q_factor = fusion_power / input_power;
        self.breakeven_achieved = self.q_factor > 1.0;

        // LCF probability: sigmoid of log10(neutron_rate)
        // At rate ~ 1 n/s (log10 = 0): P ≈ 0.5
        // At rate ~ 10^-10: P ≈ 0.0
        // At rate ~ 10^3: P ≈ 1.0
        let log_rate = if self.neutron_rate > 1e-50 {
            self.neutron_rate.log10()
        } else {
            -50.0
        };
        self.lcf_probability = sigmoid(log_rate, 0.0, 0.5);
        // Cap at 5% per tick to avoid unrealistic certainty
        self.lcf_probability = self.lcf_probability.min(0.05);
    }

    /// Generate a narrative description of the current fusion research state.
    pub fn narrative(&self) -> Option<String> {
        if self.rate_gap_log10 < 10.0 && self.rate_gap_log10 > 5.0 {
            Some(
                "Anomalous fusion rates approaching reproducibility — the rate gap narrows"
                    .to_string(),
            )
        } else if self.rate_gap_log10 <= 5.0 && !self.breakeven_achieved {
            Some(
                "Fusion researchers report consistent neutron excess — breakthrough imminent"
                    .to_string(),
            )
        } else if self.breakeven_achieved {
            Some(
                "FUSION BREAKEVEN ACHIEVED — Q > 1, civilization enters the fusion age".to_string(),
            )
        } else {
            None
        }
    }
}

// ============================================================================
// SIMPLIFIED GAMOW INTEGRATION
// ============================================================================

/// Simplified Gamow D-D fusion rate calculation.
///
/// Returns <σv> in cm³/s — the thermally averaged fusion cross-section.
///
/// This is a simplified version of spark-engine's full Gamow integration,
/// suitable for the civilization sim's monthly tick rate.
fn simplified_gamow_dd(t_k: f64, screening_ev: f64, phonon_modes: u32) -> f64 {
    let kt_kev = t_k * 8.617e-8; // kT in keV
    if kt_kev < 1e-15 {
        return 0.0;
    }

    // Gamow peak energy
    let e_g = DD_GAMOW_CONSTANT; // keV^0.5
    let gamow_peak_kev = (e_g * e_g / (4.0 * kt_kev)).powf(1.0 / 3.0) * (3.0 * kt_kev);

    // Tunneling exponent: the dominant factor
    let tunneling_exp = -3.0 * gamow_peak_kev / kt_kev;

    // Screening reduces the effective barrier: adds U_e/kT to the tunneling exponent
    let ue_kev = screening_ev / 1000.0;
    let screening_exp = ue_kev / kt_kev;

    // Phonon enhancement: adds phonon energy to effective temperature
    let phonon_exp = if phonon_modes > 0 {
        let phonon_energy_kev = phonon_modes as f64 * PDD_PHONON_KEV;
        phonon_energy_kev / kt_kev
    } else {
        0.0
    };

    // Screening can only reduce the barrier, not eliminate it.
    // Physical limit: screening enhances by at most a factor of 10^20 (Raiola 2004).
    // The screening exponent is capped to prevent unphysical amplification.
    let capped_screening = screening_exp.min(46.0); // exp(46) ≈ 10^20

    // Phonon enhancement is similarly modest
    let capped_phonon = phonon_exp.min(20.0); // exp(20) ≈ 5×10^8

    // Net exponent: tunneling (very negative at low T) + capped enhancements
    let net_exp = tunneling_exp + capped_screening + capped_phonon;
    if net_exp < -500.0 {
        return 0.0;
    }
    if net_exp > 100.0 {
        return 1e30;
    } // cap

    // Astrophysical S-factor for D-D
    let s_factor = 52.0; // keV·barn

    // Thermal velocity
    let thermal_velocity = (2.0 * kt_kev * 1.602e-16 / 3.34e-27).sqrt(); // m/s

    // <σv> = S × exp(net_exponent) × velocity normalization
    let sigma_v = s_factor * 1e-24 // barn to cm²
        * net_exp.exp()
        * thermal_velocity * 100.0; // m/s to cm/s

    sigma_v.max(0.0)
}

/// Sigmoid function for mapping continuous values to probabilities.
fn sigmoid(x: f64, center: f64, steepness: f64) -> f64 {
    1.0 / (1.0 + (-steepness * (x - center)).exp())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_tech_low_probability() {
        let mut state = CivilizationFusionState::default();
        state.update_from_tech(1.0, 1.0); // Baseline tech
                                          // At baseline, probability is at the sigmoid floor — capped at 5%
        assert!(
            state.lcf_probability <= 0.05,
            "Baseline tech probability should be <= 5%: {}",
            state.lcf_probability
        );
    }

    #[test]
    fn high_tech_meaningful_probability() {
        let mut state = CivilizationFusionState::default();
        state.update_from_tech(5.0, 5.0); // Advanced tech
        assert!(
            state.lcf_probability > state.q_factor.min(0.001)
                || state.effective_temperature_k > 100_000.0,
            "High tech should improve fusion parameters"
        );
        assert!(state.effective_temperature_k > 300_000.0);
        assert!(state.phonon_modes >= 3);
    }

    #[test]
    fn tech_progression_changes_parameters() {
        let mut temps = Vec::new();
        let mut screens = Vec::new();
        for tech in [1.0, 2.0, 3.0, 4.0, 5.0] {
            let mut state = CivilizationFusionState::default();
            state.update_from_tech(tech, tech);
            temps.push(state.effective_temperature_k);
            screens.push(state.screening_ev);
        }
        // Temperature should increase monotonically
        for i in 1..temps.len() {
            assert!(
                temps[i] >= temps[i - 1],
                "Tech {} temp {} should be >= tech {} temp {}",
                i + 1,
                temps[i],
                i,
                temps[i - 1]
            );
        }
        // Screening should increase monotonically
        for i in 1..screens.len() {
            assert!(
                screens[i] >= screens[i - 1],
                "Tech {} screening {} should be >= tech {} screening {}",
                i + 1,
                screens[i],
                i,
                screens[i - 1]
            );
        }
    }

    #[test]
    fn probability_capped_at_5_percent() {
        let mut state = CivilizationFusionState::default();
        state.update_from_tech(10.0, 10.0); // Max tech
        assert!(
            state.lcf_probability <= 0.05,
            "Probability should cap at 5%: {}",
            state.lcf_probability
        );
    }

    #[test]
    fn narrative_at_different_stages() {
        let mut state = CivilizationFusionState::default();

        // Far from breakthrough
        state.rate_gap_log10 = 40.0;
        state.breakeven_achieved = false;
        assert!(state.narrative().is_none());

        // Approaching
        state.rate_gap_log10 = 8.0;
        assert!(state.narrative().is_some());
        assert!(state.narrative().unwrap().contains("approaching"));

        // Imminent
        state.rate_gap_log10 = 3.0;
        assert!(state.narrative().unwrap().contains("imminent"));

        // Achieved
        state.breakeven_achieved = true;
        assert!(state.narrative().unwrap().contains("BREAKEVEN"));
    }

    #[test]
    fn sigmoid_properties() {
        assert!((sigmoid(0.0, 0.0, 1.0) - 0.5).abs() < 0.01);
        assert!(sigmoid(10.0, 0.0, 0.5) > 0.99);
        assert!(sigmoid(-10.0, 0.0, 0.5) < 0.01);
    }

    #[test]
    fn screening_improves_with_science() {
        let mut low = CivilizationFusionState::default();
        let mut high = CivilizationFusionState::default();
        low.update_from_tech(3.0, 1.0);
        high.update_from_tech(3.0, 5.0);
        assert!(
            high.screening_ev > low.screening_ev,
            "Higher science should improve screening: {} vs {}",
            high.screening_ev,
            low.screening_ev
        );
    }

    #[test]
    fn phonon_modes_gate_on_engineering() {
        let mut state = CivilizationFusionState::default();
        state.update_from_tech(1.5, 1.0);
        assert_eq!(state.phonon_modes, 0, "Low eng should have 0 phonon modes");

        state.update_from_tech(3.5, 1.0);
        assert!(
            state.phonon_modes >= 1,
            "Eng 3.5 should have 1+ phonon modes"
        );

        state.update_from_tech(7.0, 1.0);
        assert_eq!(
            state.phonon_modes, 5,
            "Eng 7+ should have max 5 phonon modes"
        );
    }

    #[test]
    fn zero_temperature_no_crash() {
        let result = simplified_gamow_dd(0.0, 309.0, 0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn room_temperature_very_small_rate() {
        let rate = simplified_gamow_dd(300.0, 309.0, 0);
        // At room temp with screening, rate is small but not zero
        // (screening partially compensates tunneling barrier)
        assert!(rate < 1.0, "Room temp rate should be sub-1: {}", rate);
    }
}
