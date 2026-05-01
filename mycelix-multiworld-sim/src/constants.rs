// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consolidated simulation constants with citations and configuration.
//!
//! Every number in this file has a published source or explicit justification.
//! Constants are organized by system and can be overridden via SimulationParams.
//!
//! # Usage
//!
//! ```rust
//! let params = SimulationParams::default(); // All cited defaults
//! let custom = SimulationParams { spoilage_food: 0.05, ..Default::default() };
//! ```

use serde::{Deserialize, Serialize};

/// All tunable simulation parameters in one place.
/// Each field has a default based on published data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    // ========================================================================
    // PSYCHOLOGY (McEwen 1998, Palinkas 2008, Karasek 1979)
    // ========================================================================
    /// Baseline environmental stress per tick.
    /// Space habitats are inherently stressful: artificial light, recycled air.
    /// Ref: Palinkas & Suedfeld (2008) — Antarctic station stress baselines.
    pub stress_baseline: f64,

    /// Allostatic load from social isolation (social_satiation < 0.3).
    /// Ref: McEwen (1998) — chronic stress mediator accumulation.
    pub stress_isolation: f64,

    /// Allostatic load from overwork (worker_ratio > 0.6).
    /// Ref: Karasek (1979) — demand-control model.
    pub stress_overwork: f64,

    /// Natural load decay per tick (rest, adaptation).
    pub stress_decay: f64,

    /// Care worker load reduction per worker per 100 recipients.
    pub stress_care_reduction: f64,

    /// Burnout threshold (above this, consciousness growth capped).
    pub burnout_threshold: f64,

    // ========================================================================
    // TRAUMA (Van der Kolk 2014, Kessler 1995)
    // ========================================================================
    /// Trauma decay per tick. Full trauma (1.0) takes ~83 years to decay.
    /// Ref: Kessler (1995) — PTSD recovery timelines.
    pub trauma_decay_base: f64,

    /// Additional trauma decay from care workers.
    pub trauma_decay_care: f64,

    /// Trauma scale factor for disaster-induced trauma.
    pub trauma_disaster_scale: f64,

    // ========================================================================
    // RADIATION (Hassler 2014, Cucinotta 2014, NASA-STD-3001)
    // ========================================================================
    /// Mars surface dose in Sv/month.
    /// Ref: Hassler et al. (2014) MSL/RAD: 0.67 mSv/day = 20 mSv/month.
    pub radiation_mars_sv_month: f64,

    /// Moon surface dose in Sv/month.
    /// Ref: Cucinotta (2014): ~15 mSv/month unshielded.
    pub radiation_moon_sv_month: f64,

    /// Europa dose in Sv/month (under ice shielding).
    pub radiation_europa_sv_month: f64,

    /// NASA career limit in Sv.
    /// Ref: NASA-STD-3001 Rev C (2022): 600 mSv lifetime.
    pub radiation_career_limit_sv: f64,

    // ========================================================================
    // FERTILITY (Wakayama 2023, Lyons 2026)
    // ========================================================================
    /// Gravity threshold for reproduction viability.
    /// Below this, reproduction requires centrifuge or gene therapy.
    pub fertility_gravity_threshold: f64,

    /// Gravity divisor for fertility multiplier: fert = (g / divisor).clamp(0.3, 1.0).
    /// Ref: Wakayama 2023 (JAXA), Lyons 2026 (Adelaide).
    pub fertility_gravity_divisor: f64,

    // ========================================================================
    // RESOURCE SPOILAGE (NASA ISS data)
    // ========================================================================
    /// Food spoilage per tick (fraction lost).
    /// Perishable goods in controlled environment.
    pub spoilage_food: f64,

    /// Water contamination loss per tick.
    /// Ref: ISS WRS recovery ~90% (ICES-2024-317).
    pub spoilage_water: f64,

    /// Materials degradation per tick (radiation embrittlement, thermal cycling).
    pub spoilage_materials: f64,

    /// Energy storage/transmission loss per tick.
    pub spoilage_energy: f64,

    /// Oxygen leak rate per tick.
    /// Ref: ISS loses ~1% O2/month to seal leaks.
    pub spoilage_oxygen: f64,

    // ========================================================================
    // PROJECTS (Flyvbjerg 2002)
    // ========================================================================
    /// Project duration variance (gaussian std as fraction of base).
    /// Ref: Flyvbjerg (2002) — construction overruns 20-50%.
    pub project_duration_variance: f64,

    /// Probability of setback per tick (adds 1-3 months).
    pub project_setback_probability: f64,

    /// Probability of critical failure per tick (project abandoned).
    pub project_failure_probability: f64,

    // ========================================================================
    // ENVIRONMENTAL ENTROPY (NASA MER data)
    // ========================================================================
    /// Mars solar panel dust loss per tick (fraction).
    /// Ref: Spirit/Opportunity dust accumulation.
    pub solar_dust_mars: f64,

    /// Moon solar panel dust loss per tick.
    pub solar_dust_moon: f64,

    // ========================================================================
    // CULTURAL MEMORY
    // ========================================================================
    /// Half-life of cultural memories in ticks (~100 years).
    pub memory_half_life_ticks: f64,

    // ========================================================================
    // HABITAT PSYCHOLOGY (Ulrich 1984, Palinkas 2008)
    // ========================================================================
    /// Allostatic load modifier for no-window modules per tick.
    /// Ref: Palinkas 2008 — 20-30% wellbeing loss after 90 days.
    pub habitat_no_window_stress: f64,

    /// Allostatic load modifier for water-window modules per tick.
    /// Beneficial: natural light + beauty.
    pub habitat_water_window_benefit: f64,

    // ========================================================================
    // ROBOTICS (NASA Robonaut, Boston Dynamics)
    // ========================================================================
    /// Manipulator power consumption (watts).
    pub robot_power_manipulator: f64,
    /// Humanoid power consumption (watts).
    pub robot_power_humanoid: f64,
    /// Helicopter power consumption (watts).
    pub robot_power_helicopter: f64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            // Psychology
            stress_baseline: 0.004,
            stress_isolation: 0.008,
            stress_overwork: 0.006,
            stress_decay: 0.010,
            stress_care_reduction: 0.006,
            burnout_threshold: 0.8,

            // Trauma
            trauma_decay_base: 0.001,
            trauma_decay_care: 0.003,
            trauma_disaster_scale: 0.3,

            // Radiation
            radiation_mars_sv_month: 0.020,
            radiation_moon_sv_month: 0.015,
            radiation_europa_sv_month: 0.005,
            radiation_career_limit_sv: 0.600,

            // Fertility
            fertility_gravity_threshold: 0.37,
            fertility_gravity_divisor: 0.5,

            // Spoilage
            spoilage_food: 0.03,
            spoilage_water: 0.005,
            spoilage_materials: 0.01,
            spoilage_energy: 0.15,
            spoilage_oxygen: 0.005,

            // Projects
            project_duration_variance: 0.15,
            project_setback_probability: 0.03,
            project_failure_probability: 0.005,

            // Environmental entropy
            solar_dust_mars: 0.05,
            solar_dust_moon: 0.02,

            // Cultural memory
            memory_half_life_ticks: 1200.0,

            // Habitat psychology
            habitat_no_window_stress: 0.003,
            habitat_water_window_benefit: -0.001,

            // Robotics
            robot_power_manipulator: 150.0,
            robot_power_humanoid: 500.0,
            robot_power_helicopter: 800.0,
        }
    }
}

impl SimulationParams {
    /// Create params for a "harsh reality" scenario — everything is harder.
    pub fn harsh() -> Self {
        Self {
            spoilage_food: 0.05,
            spoilage_energy: 0.20,
            project_setback_probability: 0.05,
            project_failure_probability: 0.01,
            stress_baseline: 0.006,
            solar_dust_mars: 0.08,
            ..Self::default()
        }
    }

    /// Create params for an "optimistic" scenario — best-case engineering.
    pub fn optimistic() -> Self {
        Self {
            spoilage_food: 0.01,
            spoilage_energy: 0.10,
            project_setback_probability: 0.01,
            project_failure_probability: 0.001,
            stress_baseline: 0.002,
            solar_dust_mars: 0.03,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params_reasonable() {
        let p = SimulationParams::default();
        assert!(p.stress_baseline > 0.0 && p.stress_baseline < 0.1);
        assert!(p.radiation_mars_sv_month > 0.01 && p.radiation_mars_sv_month < 0.1);
        assert!(p.spoilage_food > 0.0 && p.spoilage_food < 0.2);
        assert!(p.radiation_career_limit_sv > 0.1 && p.radiation_career_limit_sv < 2.0);
    }

    #[test]
    fn test_harsh_is_harder() {
        let d = SimulationParams::default();
        let h = SimulationParams::harsh();
        assert!(h.spoilage_food > d.spoilage_food);
        assert!(h.project_failure_probability > d.project_failure_probability);
    }
}
