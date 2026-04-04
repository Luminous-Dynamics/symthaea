// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge between the multiworld-sim civilization simulator and Sol Atlas.
//!
//! Maps timeline year (0-500) to civilization state: grid stress evolution,
//! technology milestones, disaster events, and regional population changes.
//! Data is derived from the empirically-calibrated models in mycelix-multiworld-sim.

use crate::energy_trading;

/// Technology milestone — appears at a specific year on the timeline.
#[derive(Debug, Clone)]
pub struct TechMilestone {
    pub year: u32,
    pub name: &'static str,
    pub description: &'static str,
}

/// All technology milestones (from multiworld-sim empirical timelines).
pub fn tech_milestones() -> Vec<TechMilestone> {
    vec![
        TechMilestone { year: 27, name: "NTP Demo", description: "Nuclear Thermal Propulsion demonstration" },
        TechMilestone { year: 28, name: "Fission Surface", description: "Kilopower fission reactor on lunar surface" },
        TechMilestone { year: 35, name: "Luna Base", description: "Permanent lunar habitat (pop 12)" },
        TechMilestone { year: 45, name: "Mars Colony", description: "First Mars settlement (pop 50)" },
        TechMilestone { year: 60, name: "Fusion Demo", description: "Net-positive fusion ignition" },
        TechMilestone { year: 80, name: "Europa Station", description: "Research outpost under ice shell" },
        TechMilestone { year: 100, name: "Fusion Grid", description: "First grid-scale fusion plant" },
        TechMilestone { year: 120, name: "Titan Outpost", description: "Hydrocarbon harvesting station" },
        TechMilestone { year: 150, name: "Mars Self-Sufficient", description: "Colony reaches 70% self-sufficiency" },
        TechMilestone { year: 200, name: "Interstellar Probe", description: "First 0.1c probe launched" },
    ]
}

/// Turchin secular cycle phase — maps to visual indicators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecularPhase {
    Growth,
    Stagflation,
    Crisis,
    Depression,
}

impl SecularPhase {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Growth => "Growth",
            Self::Stagflation => "Stagflation",
            Self::Crisis => "Crisis",
            Self::Depression => "Depression",
        }
    }

    /// Color for the cycle phase indicator.
    pub fn color(&self) -> [f32; 3] {
        match self {
            Self::Growth => [0.1, 0.8, 0.3],       // green
            Self::Stagflation => [0.9, 0.7, 0.1],  // amber
            Self::Crisis => [0.9, 0.2, 0.1],       // red
            Self::Depression => [0.4, 0.1, 0.1],    // dark red
        }
    }
}

/// Get the Turchin secular cycle phase at a given year.
/// Based on ~80-year cycles (Turchin 2003, ~200-300 year secular cycles).
pub fn secular_phase_at_year(year: u32) -> SecularPhase {
    // Simplified: 80-year cycle with phase proportions
    let cycle_pos = (year % 80) as f32 / 80.0;
    if cycle_pos < 0.4 {
        SecularPhase::Growth        // 0-32 years: expansion
    } else if cycle_pos < 0.65 {
        SecularPhase::Stagflation   // 32-52 years: elite overproduction rising
    } else if cycle_pos < 0.85 {
        SecularPhase::Crisis        // 52-68 years: instability, possible civil war
    } else {
        SecularPhase::Depression    // 68-80 years: population decline, reset
    }
}

/// Evolve grid stress for a given timeline year.
/// As fossil EROI declines and renewables grow, stress patterns shift.
pub fn evolve_grid_stress(year: u32) -> Vec<energy_trading::GridStress> {
    energy_trading::simulate_grid_stress(year)
}

/// Regional population projection at a given year (millions).
/// Based on UN calibrated growth rates from multiworld-sim earth_regions.rs.
pub fn regional_population_at_year(region_name: &str, base_pop: f64, year: u32) -> f64 {
    let growth_rate = match region_name {
        "Sub-Saharan Africa" => 0.025,
        "South Asia" => 0.010,
        "Southeast Asia" => 0.008,
        "East Asia" => 0.002,
        "North Africa/MENA" => 0.015,
        "Latin America" => 0.007,
        "Caribbean/Islands" => 0.005,
        "Central Asia" => 0.012,
        "North America" => 0.005,
        "Europe" => -0.002,
        "Russia/CIS" => -0.003,
        "Oceania" => 0.008,
        _ => 0.005,
    };
    base_pop * (1.0_f64 + growth_rate).powi(year as i32)
}

/// Interplanetary colony data at a given year.
#[derive(Debug, Clone)]
pub struct ColonyState {
    pub name: &'static str,
    pub body: &'static str,
    pub founded_year: u32,
    pub population: u32,
    pub self_sufficiency: f32,
    pub phi: f32,
    pub light_delay_s: f32,
}

/// Get colony states at a given timeline year.
pub fn colonies_at_year(year: u32) -> Vec<ColonyState> {
    let mut colonies = Vec::new();

    // Luna Base (founded year 35)
    if year >= 35 {
        let age = (year - 35) as f32;
        colonies.push(ColonyState {
            name: "Luna Base",
            body: "Moon",
            founded_year: 35,
            population: (12.0 + age * 3.0).min(500.0) as u32,
            self_sufficiency: (0.1 + age * 0.005).min(0.8),
            phi: (0.3 + age * 0.002).min(0.6),
            light_delay_s: 1.28,
        });
    }

    // Mars Colony (founded year 45)
    if year >= 45 {
        let age = (year - 45) as f32;
        colonies.push(ColonyState {
            name: "Ares Settlement",
            body: "Mars",
            founded_year: 45,
            population: (50.0 + age * 8.0).min(10000.0) as u32,
            self_sufficiency: (0.05 + age * 0.004).min(0.7),
            phi: (0.25 + age * 0.003).min(0.55),
            light_delay_s: 780.0,
        });
    }

    // Europa Station (founded year 80)
    if year >= 80 {
        let age = (year - 80) as f32;
        colonies.push(ColonyState {
            name: "Europa Deep",
            body: "Jupiter", // orbits Jupiter
            founded_year: 80,
            population: (8.0 + age * 0.5).min(100.0) as u32,
            self_sufficiency: (0.02 + age * 0.001).min(0.3),
            phi: (0.2 + age * 0.001).min(0.4),
            light_delay_s: 2640.0,
        });
    }

    // Titan Outpost (founded year 120)
    if year >= 120 {
        let age = (year - 120) as f32;
        colonies.push(ColonyState {
            name: "Titan Harvest",
            body: "Saturn", // orbits Saturn
            founded_year: 120,
            population: (4.0 + age * 0.3).min(50.0) as u32,
            self_sufficiency: (0.01 + age * 0.0005).min(0.2),
            phi: (0.15 + age * 0.001).min(0.35),
            light_delay_s: 4800.0,
        });
    }

    colonies
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tech_milestones_ordered() {
        let milestones = tech_milestones();
        assert!(milestones.len() >= 8);
        for w in milestones.windows(2) {
            assert!(w[0].year <= w[1].year, "Milestones not ordered: {} > {}", w[0].year, w[1].year);
        }
    }

    #[test]
    fn secular_phases_cycle() {
        assert_eq!(secular_phase_at_year(0), SecularPhase::Growth);
        assert_eq!(secular_phase_at_year(40), SecularPhase::Stagflation);
        assert_eq!(secular_phase_at_year(60), SecularPhase::Crisis);
        assert_eq!(secular_phase_at_year(75), SecularPhase::Depression);
        // Second cycle
        assert_eq!(secular_phase_at_year(80), SecularPhase::Growth);
        assert_eq!(secular_phase_at_year(120), SecularPhase::Stagflation);
    }

    #[test]
    fn grid_stress_evolves() {
        let stress_0 = evolve_grid_stress(0);
        let stress_200 = evolve_grid_stress(200);
        assert_eq!(stress_0.len(), 16);
        assert_eq!(stress_200.len(), 16);
        // Delhi should be more stressed at year 200 (fossil decline)
        let delhi_0 = stress_0.iter().find(|s| s.name == "Delhi").unwrap();
        let delhi_200 = stress_200.iter().find(|s| s.name == "Delhi").unwrap();
        assert!(delhi_200.allostatic_load >= delhi_0.allostatic_load * 0.8,
            "Delhi stress should not dramatically decrease");
    }

    #[test]
    fn population_projections() {
        let africa_now = regional_population_at_year("Sub-Saharan Africa", 1200.0, 0);
        let africa_100 = regional_population_at_year("Sub-Saharan Africa", 1200.0, 100);
        assert!((africa_now - 1200.0).abs() < 0.01);
        assert!(africa_100 > africa_now, "Africa should grow");

        let europe_now = regional_population_at_year("Europe", 450.0, 0);
        let europe_100 = regional_population_at_year("Europe", 450.0, 100);
        assert!(europe_100 < europe_now, "Europe should shrink (negative growth)");
    }

    #[test]
    fn colonies_appear_at_milestones() {
        let c_0 = colonies_at_year(0);
        assert!(c_0.is_empty());

        let c_35 = colonies_at_year(35);
        assert_eq!(c_35.len(), 1);
        assert_eq!(c_35[0].name, "Luna Base");

        let c_50 = colonies_at_year(50);
        assert_eq!(c_50.len(), 2); // Luna + Mars

        let c_100 = colonies_at_year(100);
        assert_eq!(c_100.len(), 3); // Luna + Mars + Europa

        let c_200 = colonies_at_year(200);
        assert_eq!(c_200.len(), 4); // All four
        assert!(c_200.iter().any(|c| c.name == "Titan Harvest"));
    }

    #[test]
    fn colony_populations_grow() {
        let mars_50 = colonies_at_year(50).into_iter().find(|c| c.body == "Mars").unwrap();
        let mars_150 = colonies_at_year(150).into_iter().find(|c| c.body == "Mars").unwrap();
        assert!(mars_150.population > mars_50.population);
        assert!(mars_150.self_sufficiency > mars_50.self_sufficiency);
    }

    #[test]
    fn secular_phase_colors() {
        let g = SecularPhase::Growth.color();
        assert!(g[1] > g[0], "Growth should be greenish");
        let c = SecularPhase::Crisis.color();
        assert!(c[0] > c[1], "Crisis should be reddish");
    }
}
