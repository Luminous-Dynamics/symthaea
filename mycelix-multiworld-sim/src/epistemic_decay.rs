// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Epistemic decay: SKILL INTEGRITY, not data corruption.
//!
//! Stored data on Holochain is effectively immortal (cryptographic hashing +
//! redundancy across 667 source chains). The ship's library doesn't decay.
//!
//! What DOES decay is the ability to USE that data. Can someone read the
//! reactor manual and actually fix the reactor? That depends on:
//!
//! 1. Skill attrition: unused skills decay (see skill_integrity.rs)
//! 2. Henrich cultural attrition: small populations can't transmit complex skills
//! 3. Comprehension gap: stored data requires base skills to understand
//!
//! Data is immortal. Understanding is mortal. That's the real horror.
//!
//! # References
//!
//! - Henrich, J. (2004). Demography and cultural evolution. American Antiquity.
//! - Boyd & Richerson (1985). Culture and the Evolutionary Process.
//! - Tasmania effect: isolated small populations lose complex technology.

use serde::{Deserialize, Serialize};

/// Henrich threshold: below this population, complex skills can't be transmitted.
const HENRICH_THRESHOLD: usize = 200;
/// Base radiation decay rate per Sievert per tick.
const RADIATION_DECAY_PER_SV: f64 = 0.0001;
/// Cosmic ray multiplier outside heliosphere.
const INTERSTELLAR_RAY_MULT: f64 = 3.0;
/// Energy cost per unit of knowledge integrity maintained per tick (abstract).
const CORRECTION_ENERGY_PER_UNIT: f64 = 0.1;

/// Knowledge integrity state for a closed system (generation ship).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicState {
    /// Knowledge integrity per sector [0, 1] (8 sectors).
    /// 1.0 = full knowledge, 0.0 = completely forgotten.
    pub sector_integrity: [f64; 8],
    /// Energy cost of error correction per tick.
    pub correction_energy_cost: f64,
    /// Whether error correction is currently active.
    pub correction_active: bool,
    /// Total knowledge lost (cumulative, for reporting).
    pub cumulative_loss: f64,
    /// Number of ticks correction has been off.
    pub correction_off_ticks: u32,
}

impl Default for EpistemicState {
    fn default() -> Self {
        Self {
            sector_integrity: [1.0; 8],
            correction_energy_cost: 0.0,
            correction_active: true,
            cumulative_loss: 0.0,
            correction_off_ticks: 0,
        }
    }
}

impl EpistemicState {
    /// Tick the epistemic decay model.
    ///
    /// # Arguments
    /// - `population`: Current living population
    /// - `cosmic_ray_dose_sv`: Accumulated radiation dose (Sieverts)
    /// - `outside_heliosphere`: Whether the ship has left the heliosphere
    /// - `energy_available`: Energy budget fraction available for correction [0, 1]
    /// - `tech_levels`: Current technology levels per sector (from embodied knowledge)
    pub fn tick(
        &mut self,
        population: usize,
        cosmic_ray_dose_sv: f64,
        outside_heliosphere: bool,
        energy_available: f64,
        tech_levels: &[f64; 8],
    ) {
        // 1. Radiation bit-flip decay
        let ray_mult = if outside_heliosphere {
            INTERSTELLAR_RAY_MULT
        } else {
            1.0
        };
        let radiation_decay = RADIATION_DECAY_PER_SV * cosmic_ray_dose_sv * ray_mult;

        // 2. Henrich cultural attrition
        let henrich_decay = if population < HENRICH_THRESHOLD {
            let deficit = (HENRICH_THRESHOLD - population) as f64 / HENRICH_THRESHOLD as f64;
            deficit * 0.01 // Up to 1% per tick when population is zero
        } else {
            0.0
        };

        // 3. Error correction trade-off
        let total_knowledge: f64 = self.sector_integrity.iter().sum();
        self.correction_energy_cost = total_knowledge * CORRECTION_ENERGY_PER_UNIT;

        // Decide whether to run error correction based on energy budget
        self.correction_active = energy_available > 0.3; // Need at least 30% energy budget

        let correction_factor = if self.correction_active {
            0.1 // Error correction blocks 90% of decay
        } else {
            self.correction_off_ticks += 1;
            1.0 // Full decay without correction
        };
        if self.correction_active {
            self.correction_off_ticks = 0;
        }

        // Apply decay to each sector
        for i in 0..8 {
            let total_decay = (radiation_decay + henrich_decay) * correction_factor;

            // Higher tech levels are harder to maintain (more complex = more fragile)
            let complexity_mult = 1.0 + (tech_levels[i] - 1.0).max(0.0) * 0.2;

            let sector_decay = total_decay * complexity_mult;
            self.sector_integrity[i] = (self.sector_integrity[i] - sector_decay).max(0.0);
            self.cumulative_loss += sector_decay;
        }
    }

    /// Mean knowledge integrity across all sectors [0, 1].
    pub fn mean_integrity(&self) -> f64 {
        self.sector_integrity.iter().sum::<f64>() / 8.0
    }

    /// Which sector has the lowest integrity?
    pub fn weakest_sector(&self) -> (usize, f64) {
        let mut min_idx = 0;
        let mut min_val = f64::MAX;
        for (i, &v) in self.sector_integrity.iter().enumerate() {
            if v < min_val {
                min_val = v;
                min_idx = i;
            }
        }
        (min_idx, min_val)
    }

    /// Whether any sector has dropped below 50% integrity (critical knowledge loss).
    pub fn has_critical_loss(&self) -> bool {
        self.sector_integrity.iter().any(|&v| v < 0.5)
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        let (weak_idx, weak_val) = self.weakest_sector();
        let sector_names = ["Eng", "Agr", "Med", "Gov", "Sci", "Edu", "Art", "Log"];
        format!(
            "Integrity: {:.1}% | Weakest: {} ({:.1}%) | Correction: {} | Lost: {:.3}",
            self.mean_integrity() * 100.0,
            sector_names[weak_idx],
            weak_val * 100.0,
            if self.correction_active { "ON" } else { "OFF" },
            self.cumulative_loss,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_at_full_integrity() {
        let state = EpistemicState::default();
        assert!((state.mean_integrity() - 1.0).abs() < 1e-10);
        assert!(!state.has_critical_loss());
    }

    #[test]
    fn radiation_causes_decay() {
        let mut state = EpistemicState::default();
        let tech = [2.0; 8];
        // High radiation, outside heliosphere
        for _ in 0..120 {
            // 10 years
            state.tick(500, 3.0, true, 1.0, &tech);
        }
        assert!(
            state.mean_integrity() < 1.0,
            "Radiation should cause decay: {:.3}",
            state.mean_integrity()
        );
        assert!(
            state.mean_integrity() > 0.5,
            "10 years shouldn't destroy everything: {:.3}",
            state.mean_integrity()
        );
    }

    #[test]
    fn henrich_effect_below_threshold() {
        let mut state = EpistemicState::default();
        let tech = [1.0; 8];
        // Small population, minimal radiation
        for _ in 0..120 {
            // 10 years
            state.tick(50, 0.1, false, 1.0, &tech);
        }
        assert!(
            state.mean_integrity() < 1.0,
            "Small pop should cause knowledge loss: {:.3}",
            state.mean_integrity()
        );
    }

    #[test]
    fn no_decay_with_large_pop_and_low_radiation() {
        let mut state = EpistemicState::default();
        let tech = [1.0; 8];
        // Large population, no radiation, inside heliosphere
        for _ in 0..12 {
            state.tick(500, 0.0, false, 1.0, &tech);
        }
        assert!(
            (state.mean_integrity() - 1.0).abs() < 0.01,
            "No decay expected: {:.3}",
            state.mean_integrity()
        );
    }

    #[test]
    fn correction_off_accelerates_decay() {
        let mut state_on = EpistemicState::default();
        let mut state_off = EpistemicState::default();
        let tech = [2.0; 8];

        for _ in 0..60 {
            // 5 years
            state_on.tick(500, 2.0, true, 1.0, &tech); // Correction ON (energy > 0.3)
            state_off.tick(500, 2.0, true, 0.1, &tech); // Correction OFF (energy < 0.3)
        }

        assert!(
            state_off.mean_integrity() < state_on.mean_integrity(),
            "Correction OFF should decay faster: {:.3} vs {:.3}",
            state_off.mean_integrity(),
            state_on.mean_integrity()
        );
    }

    #[test]
    fn high_tech_decays_faster() {
        let mut state_low = EpistemicState::default();
        let mut state_high = EpistemicState::default();
        let tech_low = [1.0; 8];
        let tech_high = [5.0; 8];

        for _ in 0..60 {
            state_low.tick(500, 2.0, true, 1.0, &tech_low);
            state_high.tick(500, 2.0, true, 1.0, &tech_high);
        }

        assert!(
            state_high.mean_integrity() < state_low.mean_integrity(),
            "High tech should be more fragile: {:.3} vs {:.3}",
            state_high.mean_integrity(),
            state_low.mean_integrity()
        );
    }

    #[test]
    fn critical_loss_detected() {
        let mut state = EpistemicState::default();
        state.sector_integrity[4] = 0.4; // Science dropped below 50%
        assert!(state.has_critical_loss());
    }

    #[test]
    fn summary_produces_string() {
        let state = EpistemicState::default();
        let s = state.summary();
        assert!(s.contains("Integrity:"));
        assert!(s.contains("100.0%"));
    }
}
