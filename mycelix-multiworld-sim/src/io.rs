// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! TOML input / JSON output for the simulation.
//!
//! TOML for configuration (human-readable, editable).
//! JSON for results (machine-readable, analyzable in Python/R).
//!
//! ```
//! scenario.toml → SimulationConfig + PolicyConfig + SimulationParams
//!                                ↓
//!                         run simulation
//!                                ↓
//!                         output.json (structured results)
//! ```

use crate::config::{BirthPolicy, ProjectStrategy, ResourcePriority, SimulationConfig};
use crate::constants::SimulationParams;
use serde::{Deserialize, Serialize};

/// TOML-friendly scenario configuration.
/// All fields optional — defaults from SimulationConfig::default_300_year().
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ScenarioToml {
    pub simulation: Option<SimToml>,
    pub policy: Option<PolicyToml>,
    pub params: Option<ParamsToml>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SimToml {
    pub years: Option<u32>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PolicyToml {
    pub birth_policy: Option<String>,
    pub project_strategy: Option<String>,
    pub resource_priority: Option<String>,
    pub trade_openness: Option<f64>,
    pub defense_spending: Option<f64>,
    pub exploration_investment: Option<f64>,
    pub trust_weighted_governance: Option<bool>,
    pub disasters_enabled: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ParamsToml {
    pub radiation_mars_sv_month: Option<f64>,
    pub spoilage_food: Option<f64>,
    pub spoilage_energy: Option<f64>,
    pub stress_baseline: Option<f64>,
    pub trauma_decay_base: Option<f64>,
    pub project_setback_probability: Option<f64>,
    pub project_failure_probability: Option<f64>,
    pub solar_dust_mars: Option<f64>,
    pub memory_half_life_ticks: Option<f64>,
}

impl ScenarioToml {
    /// Load from a TOML file path.
    pub fn load(path: &str) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
        toml::from_str(&content).map_err(|e| format!("Failed to parse TOML: {}", e))
    }

    /// Convert to simulation config, policy, and params.
    pub fn into_config(self) -> (SimulationConfig, SimulationParams) {
        let sim = self.simulation.unwrap_or_default();
        let years = sim.years.unwrap_or(300);
        let seed = sim.seed.unwrap_or(42);

        let mut config = match years {
            y if y >= 1000 => SimulationConfig::default_1000_year(),
            y if y >= 777 => SimulationConfig::default_777_year(),
            y if y >= 300 => SimulationConfig::default_300_year(),
            _ => SimulationConfig::default_150_year(),
        };
        config.seed = seed;
        config.total_ticks = years * 12;

        // Apply policy overrides
        if let Some(pol) = self.policy {
            if let Some(bp) = pol.birth_policy {
                config.policy.birth_policy = match bp.as_str() {
                    "ProNatal" => BirthPolicy::ProNatal,
                    "PopulationControl" => BirthPolicy::PopulationControl,
                    "ReplacementOnly" => BirthPolicy::ReplacementOnly,
                    _ => BirthPolicy::Natural,
                };
            }
            if let Some(ps) = pol.project_strategy {
                config.policy.project_strategy = match ps.as_str() {
                    "SurvivalFirst" => ProjectStrategy::SurvivalFirst,
                    "GrowthFirst" => ProjectStrategy::GrowthFirst,
                    "ScienceFirst" => ProjectStrategy::ScienceFirst,
                    "IndependenceFirst" => ProjectStrategy::IndependenceFirst,
                    _ => ProjectStrategy::Balanced,
                };
            }
            if let Some(rp) = pol.resource_priority {
                config.policy.resource_priority = match rp.as_str() {
                    "Industrial" => ResourcePriority::Industrial,
                    "Biological" => ResourcePriority::Biological,
                    "Knowledge" => ResourcePriority::Knowledge,
                    _ => ResourcePriority::Balanced,
                };
            }
            if let Some(v) = pol.trade_openness {
                config.policy.trade_openness = v;
            }
            if let Some(v) = pol.defense_spending {
                config.policy.defense_spending = v;
            }
            if let Some(v) = pol.exploration_investment {
                config.policy.exploration_investment = v;
            }
            if let Some(v) = pol.trust_weighted_governance {
                config.policy.trust_weighted_governance = v;
            }
            if let Some(v) = pol.disasters_enabled {
                config.policy.disasters_enabled = v;
            }
        }

        // Apply params overrides
        let mut params = SimulationParams::default();
        if let Some(p) = self.params {
            if let Some(v) = p.radiation_mars_sv_month {
                params.radiation_mars_sv_month = v;
            }
            if let Some(v) = p.spoilage_food {
                params.spoilage_food = v;
            }
            if let Some(v) = p.spoilage_energy {
                params.spoilage_energy = v;
            }
            if let Some(v) = p.stress_baseline {
                params.stress_baseline = v;
            }
            if let Some(v) = p.trauma_decay_base {
                params.trauma_decay_base = v;
            }
            if let Some(v) = p.project_setback_probability {
                params.project_setback_probability = v;
            }
            if let Some(v) = p.project_failure_probability {
                params.project_failure_probability = v;
            }
            if let Some(v) = p.solar_dust_mars {
                params.solar_dust_mars = v;
            }
            if let Some(v) = p.memory_half_life_ticks {
                params.memory_half_life_ticks = v;
            }
        }

        (config, params)
    }
}

// ============================================================================
// JSON Output
// ============================================================================

/// Structured simulation output for machine consumption.
#[derive(Debug, Clone, Serialize)]
pub struct SimulationOutput {
    /// Configuration used for this run.
    pub seed: u64,
    pub years: u32,
    pub ticks: u32,
    /// Final state.
    pub survived: bool,
    pub final_cvs: f64,
    pub final_population: usize,
    pub tech_milestones: usize,
    /// CVS trajectory (sampled every 50 years).
    pub cvs_trajectory: Vec<TrajectoryPoint>,
    /// Per-world final state.
    pub worlds: Vec<WorldOutput>,
    /// Key events.
    pub milestones: Vec<String>,
    pub narrative_event_count: usize,
    pub robot_count: usize,
    pub brownout_count: usize,
    pub exploration_count: usize,
    pub independence_count: usize,
    pub total_disasters: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrajectoryPoint {
    pub year: f64,
    pub population: usize,
    pub phi: f64,
    pub cvs: f64,
    pub trauma: f64,
    pub factions: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorldOutput {
    pub name: String,
    pub location: String,
    pub population: usize,
    pub phi: f64,
    pub conatus: f64,
    pub load: f64,
    pub self_sufficiency: f64,
    pub robot_count: usize,
}

/// Build structured output from a completed simulation.
pub fn build_output(
    sim: &crate::MultiWorldSimulator,
    report: &crate::report::CivilizationReport,
) -> SimulationOutput {
    let worlds: Vec<WorldOutput> = sim
        .worlds
        .iter()
        .map(|w| {
            let living: Vec<_> = w.agents.iter().filter(|a| a.is_alive()).collect();
            let mean_conatus = if living.is_empty() {
                0.0
            } else {
                living
                    .iter()
                    .map(|a| a.needs.affect.net_conatus())
                    .sum::<f64>()
                    / living.len() as f64
            };
            WorldOutput {
                name: w.name.clone(),
                location: w.location.clone(),
                population: w.population(),
                phi: w.collective_phi(),
                conatus: mean_conatus,
                load: {
                    let living: Vec<_> = w.agents.iter().filter(|a| a.is_alive()).collect();
                    if living.is_empty() {
                        0.0
                    } else {
                        living.iter().map(|a| a.needs.allostatic_load).sum::<f64>()
                            / living.len() as f64
                    }
                },
                self_sufficiency: w.resources.self_sufficiency(),
                robot_count: w.fleet.total_units(),
            }
        })
        .collect();

    let cvs_trajectory: Vec<TrajectoryPoint> = report
        .epoch_snapshots
        .iter()
        .filter(|s| s.tick % 600 == 0) // Every 50 years
        .map(|s| TrajectoryPoint {
            year: s.tick as f64 / 12.0,
            population: s.total_population,
            phi: s.mean_phi,
            cvs: s.civilization_viability_score,
            trauma: s.trauma_level,
            factions: s.faction_count,
        })
        .collect();

    let robot_count: usize = sim.worlds.iter().map(|w| w.fleet.total_units()).sum();
    let brownout_count = sim
        .events
        .iter()
        .filter(|e| e.description.contains("BROWNOUT"))
        .count();
    let exploration_count = sim
        .events
        .iter()
        .filter(|e| e.description.contains("EXPLORATION"))
        .count();
    let independence_count = sim
        .events
        .iter()
        .filter(|e| e.description.contains("INDEPENDENCE"))
        .count();

    SimulationOutput {
        seed: sim.config.seed,
        years: sim.config.total_ticks / 12,
        ticks: sim.config.total_ticks,
        survived: report.survived,
        final_cvs: report.final_cvs,
        final_population: report.final_population,
        tech_milestones: report.tech_milestones_achieved,
        cvs_trajectory,
        worlds,
        milestones: sim
            .events
            .iter()
            .filter(|e| e.description.contains("MILESTONE"))
            .map(|e| e.description.clone())
            .collect(),
        narrative_event_count: sim.narrative_engine.events.len(),
        robot_count,
        brownout_count,
        exploration_count,
        independence_count,
        total_disasters: report.total_disasters as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_parsing() {
        let toml_str = r#"
[simulation]
years = 500
seed = 99

[policy]
birth_policy = "ProNatal"
trade_openness = 0.3

[params]
spoilage_food = 0.05
"#;
        let scenario: ScenarioToml = toml::from_str(toml_str).unwrap();
        let (config, params) = scenario.into_config();
        assert_eq!(config.seed, 99);
        assert_eq!(config.total_ticks, 6000);
        assert!((params.spoilage_food - 0.05).abs() < 0.001);
    }
}
