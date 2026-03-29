// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Lattice Lifetime Model
//!
//! Tracks radiation damage, D/Pd ratio decay, and replacement scheduling.

use serde::{Deserialize, Serialize};

/// Lattice degradation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeState {
    /// Accumulated displacement damage (DPA)
    pub accumulated_dpa: f64,
    /// Initial D/Pd loading ratio
    pub initial_d_pd: f64,
    /// Current D/Pd loading ratio
    pub current_d_pd: f64,
    /// DPA rate (DPA/year)
    pub dpa_rate_per_year: f64,
    /// Remaining lifetime (years)
    pub remaining_years: f64,
    /// Operating hours since replacement
    pub operating_hours: f64,
    /// Annealing recovery factor (0-1)
    pub annealing_recovery: f64,
    /// Whether replacement is needed
    pub needs_replacement: bool,
}

impl Default for LatticeState {
    fn default() -> Self {
        Self::new(0.7)
    }
}

impl LatticeState {
    /// Create new lattice with initial D/Pd ratio.
    pub fn new(initial_d_pd: f64) -> Self {
        Self {
            accumulated_dpa: 0.0,
            initial_d_pd,
            current_d_pd: initial_d_pd,
            dpa_rate_per_year: 0.0,
            remaining_years: f64::INFINITY,
            operating_hours: 0.0,
            annealing_recovery: 0.0,
            needs_replacement: false,
        }
    }

    /// Typical PdD starting at D/Pd = 0.7
    pub fn pdd_typical() -> Self {
        Self::new(0.7)
    }
}

/// Lattice lifetime calculator.
pub struct LatticeLifetime;

impl LatticeLifetime {
    /// Critical DPA for Pd (void swelling onset)
    pub const CRITICAL_DPA: f64 = 10.0;

    /// D/Pd decay per DPA (~5%)
    pub const DECAY_PER_DPA: f64 = 0.05;

    /// Minimum D/Pd for viable fusion
    pub const MIN_D_PD: f64 = 0.5;

    /// Displacement threshold for Pd (eV)
    pub const DISPLACEMENT_THRESHOLD_EV: f64 = 34.0;

    /// Compute DPA rate from neutron flux.
    ///
    /// For 2.45 MeV D-D neutrons in Pd.
    pub fn compute_dpa_rate(neutron_flux_cm2_s: f64, neutron_energy_mev: f64) -> f64 {
        let e_d = Self::DISPLACEMENT_THRESHOLD_EV;

        // Elastic scattering cross-section (barn)
        let sigma_el_barn = if neutron_energy_mev < 5.0 { 3.5 } else { 2.5 };
        let sigma_el_cm2 = sigma_el_barn * 1e-24;

        // Average recoil energy
        let a_pd = 106.0;
        let t_avg_kev = 2.0 * neutron_energy_mev * 1000.0 * a_pd / (a_pd + 1.0).powi(2);
        let t_avg_ev = t_avg_kev * 1000.0;

        // Modified Kinchin-Pease
        let nu = if t_avg_ev > 2.5 * e_d {
            0.8 * t_avg_ev / (2.0 * e_d)
        } else {
            1.0
        };

        // Pd number density
        let n_pd = 12.02 * 6.022e23 / 106.42;

        // DPA rate
        let dpa_per_s = neutron_flux_cm2_s * sigma_el_cm2 * nu / n_pd;
        dpa_per_s * 3600.0 * 24.0 * 365.25
    }

    /// Update lattice state after operation.
    ///
    /// # Arguments
    /// * `state` - Current lattice state (modified in place)
    /// * `hours` - Operating hours
    /// * `neutron_rate_s` - Neutron production rate (n/s)
    /// * `surface_cm2` - Core surface area (cm²)
    /// * `temp_k` - Lattice temperature (K)
    pub fn update(
        state: &mut LatticeState,
        hours: f64,
        neutron_rate_s: f64,
        surface_cm2: f64,
        temp_k: f64,
    ) {
        state.operating_hours += hours;

        // Neutron flux
        let flux = neutron_rate_s / surface_cm2;
        state.dpa_rate_per_year = Self::compute_dpa_rate(flux, 2.45);

        // Accumulate damage
        let years = hours / (24.0 * 365.25);
        let delta_dpa = state.dpa_rate_per_year * years;

        // Annealing recovery at elevated temperature
        state.annealing_recovery = if temp_k > 400.0 {
            let kb = 8.617e-5;
            let e_a = 1.0;
            let recovery = (-e_a / (kb * temp_k)).exp() / (-e_a / (kb * 600.0)).exp();
            (recovery * 0.3).min(0.9)
        } else {
            0.0
        };

        // Net damage
        let net_dpa = delta_dpa * (1.0 - state.annealing_recovery);
        state.accumulated_dpa += net_dpa;

        // D/Pd decay
        let decay = (-Self::DECAY_PER_DPA * state.accumulated_dpa).exp();
        state.current_d_pd = state.initial_d_pd * decay;

        // Check replacement
        state.needs_replacement =
            state.current_d_pd < Self::MIN_D_PD || state.accumulated_dpa > Self::CRITICAL_DPA;

        // Remaining lifetime
        if state.dpa_rate_per_year > 0.0 && !state.needs_replacement {
            let dpa_to_critical =
                -(Self::MIN_D_PD / state.initial_d_pd).ln() / Self::DECAY_PER_DPA;
            let dpa_remaining = (dpa_to_critical - state.accumulated_dpa).max(0.0);
            state.remaining_years = dpa_remaining / state.dpa_rate_per_year;
        } else if state.needs_replacement {
            state.remaining_years = 0.0;
        }
    }

    /// Reset lattice (replacement).
    pub fn replace(state: &mut LatticeState) {
        state.accumulated_dpa = 0.0;
        state.current_d_pd = state.initial_d_pd;
        state.needs_replacement = false;
        state.operating_hours = 0.0;
        state.remaining_years = f64::INFINITY;
    }

    /// Compute replacement schedule.
    ///
    /// Returns (years_between_replacements, total_replacements).
    pub fn replacement_schedule(
        state: &LatticeState,
        neutron_rate_s: f64,
        surface_cm2: f64,
        reactor_lifetime_years: f64,
    ) -> (f64, u32) {
        let flux = neutron_rate_s / surface_cm2;
        let dpa_rate = Self::compute_dpa_rate(flux, 2.45);

        if dpa_rate <= 0.0 {
            return (f64::INFINITY, 0);
        }

        let dpa_to_critical = -(Self::MIN_D_PD / state.initial_d_pd).ln() / Self::DECAY_PER_DPA;
        let years_between = dpa_to_critical / dpa_rate;

        let total = if years_between > 0.0 {
            (reactor_lifetime_years / years_between).ceil() as u32
        } else {
            0
        };

        (years_between, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_creation() {
        let state = LatticeState::pdd_typical();
        assert_eq!(state.initial_d_pd, 0.7);
        assert_eq!(state.current_d_pd, 0.7);
        assert!(!state.needs_replacement);
    }

    #[test]
    fn test_dpa_rate() {
        let rate = LatticeLifetime::compute_dpa_rate(1e10, 2.45);
        assert!(rate > 0.0);
        assert!(rate < 1000.0); // Sanity check
    }

    #[test]
    fn test_lattice_update() {
        let mut state = LatticeState::pdd_typical();
        LatticeLifetime::update(&mut state, 1000.0, 1e10, 50.0, 500.0);

        assert!(state.operating_hours == 1000.0);
        assert!(state.dpa_rate_per_year >= 0.0);
    }

    #[test]
    fn test_replacement() {
        let mut state = LatticeState::pdd_typical();
        state.accumulated_dpa = 5.0;
        state.current_d_pd = 0.6;

        LatticeLifetime::replace(&mut state);

        assert_eq!(state.accumulated_dpa, 0.0);
        assert_eq!(state.current_d_pd, 0.7);
    }
}
