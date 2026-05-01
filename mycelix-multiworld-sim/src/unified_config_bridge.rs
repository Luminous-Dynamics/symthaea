// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge between `luminous_sim_core::UnifiedConfig` and the simulator's
//! internal `SimulationConfig` / `PolicyConfig` / `SimulationParams`.
//!
//! This allows users to provide a single TOML file (the UnifiedConfig schema)
//! which gets converted into the existing config types. Backward-compatible:
//! the existing scenario TOML files also parse correctly.

use crate::config::{
    BirthPolicy, PolicyConfig, ProjectStrategy, ResourcePriority, SimulationConfig,
};
use crate::constants::SimulationParams;
use luminous_sim_core::UnifiedConfig;

/// Convert a `UnifiedConfig` into the simulator's existing config types.
///
/// Any field that is `None` in the unified config falls back to the default
/// in the existing type. This preserves backward compatibility.
pub fn apply_unified_config(
    unified: &UnifiedConfig,
    sim_config: &mut SimulationConfig,
    policy: &mut PolicyConfig,
    params: &mut SimulationParams,
) {
    // --- Simulation section ---
    sim_config.seed = unified.simulation.seed;
    sim_config.total_ticks = unified.simulation.years * 12;

    // --- Policy section ---
    if let Some(v) = unified.policy.trust_weighted_governance {
        policy.trust_weighted_governance = v;
    }
    if let Some(v) = unified.policy.turchin_cycles_enabled {
        policy.turchin_cycles_enabled = v;
    }
    if let Some(v) = unified.policy.fep_immiseration_enabled {
        policy.fep_immiseration_enabled = v;
    }
    if let Some(v) = unified.policy.sacred_stillness_enabled {
        policy.sacred_stillness_enabled = v;
    }
    if let Some(v) = unified.policy.disasters_enabled {
        policy.disasters_enabled = v;
    }
    if let Some(v) = unified.policy.faction_enabled {
        policy.faction_enabled = v;
    }
    if let Some(v) = unified.policy.education_enabled {
        policy.education_enabled = v;
    }
    if let Some(v) = unified.policy.migration_enabled {
        policy.migration_enabled = v;
    }
    if let Some(v) = unified.policy.hybrid_earth {
        policy.hybrid_earth = v;
    }

    // Disaster sub-toggles
    if let Some(v) = unified.policy.disasters_solar_enabled {
        policy.disasters_solar_enabled = v;
    }
    if let Some(v) = unified.policy.disasters_impact_enabled {
        policy.disasters_impact_enabled = v;
    }
    if let Some(v) = unified.policy.disasters_planetary_enabled {
        policy.disasters_planetary_enabled = v;
    }
    if let Some(v) = unified.policy.disasters_eclss_enabled {
        policy.disasters_eclss_enabled = v;
    }
    if let Some(v) = unified.policy.disasters_psychological_enabled {
        policy.disasters_psychological_enabled = v;
    }
    if let Some(v) = unified.policy.disasters_technology_enabled {
        policy.disasters_technology_enabled = v;
    }
    if let Some(v) = unified.policy.disasters_civilization_enabled {
        policy.disasters_civilization_enabled = v;
    }

    // Numeric knobs
    if let Some(v) = unified.policy.pair_bond_rate {
        policy.pair_bond_rate = v;
    }
    if let Some(v) = unified.policy.care_effectiveness {
        policy.care_effectiveness = v;
    }
    if let Some(v) = unified.policy.trade_openness {
        policy.trade_openness = v;
    }
    if let Some(v) = unified.policy.defense_spending {
        policy.defense_spending = v;
    }
    if let Some(v) = unified.policy.exploration_investment {
        policy.exploration_investment = v;
    }

    // Enums
    if let Some(ref s) = unified.policy.birth_policy {
        policy.birth_policy = match s.as_str() {
            "ProNatal" => BirthPolicy::ProNatal,
            "PopulationControl" => BirthPolicy::PopulationControl,
            "ReplacementOnly" => BirthPolicy::ReplacementOnly,
            _ => BirthPolicy::Natural,
        };
    }
    if let Some(ref s) = unified.policy.project_strategy {
        policy.project_strategy = match s.as_str() {
            "SurvivalFirst" => ProjectStrategy::SurvivalFirst,
            "GrowthFirst" => ProjectStrategy::GrowthFirst,
            "ScienceFirst" => ProjectStrategy::ScienceFirst,
            "IndependenceFirst" => ProjectStrategy::IndependenceFirst,
            _ => ProjectStrategy::Balanced,
        };
    }
    if let Some(ref s) = unified.policy.resource_priority {
        policy.resource_priority = match s.as_str() {
            "Industrial" => ResourcePriority::Industrial,
            "Biological" => ResourcePriority::Biological,
            "Knowledge" => ResourcePriority::Knowledge,
            _ => ResourcePriority::Balanced,
        };
    }

    // --- Params section ---
    if let Some(v) = unified.params.stress_baseline {
        params.stress_baseline = v;
    }
    if let Some(v) = unified.params.stress_isolation {
        params.stress_isolation = v;
    }
    if let Some(v) = unified.params.stress_overwork {
        params.stress_overwork = v;
    }
    if let Some(v) = unified.params.stress_decay {
        params.stress_decay = v;
    }
    if let Some(v) = unified.params.burnout_threshold {
        params.burnout_threshold = v;
    }
    if let Some(v) = unified.params.trauma_decay_base {
        params.trauma_decay_base = v;
    }
    if let Some(v) = unified.params.trauma_disaster_scale {
        params.trauma_disaster_scale = v;
    }
    if let Some(v) = unified.params.radiation_mars_sv_month {
        params.radiation_mars_sv_month = v;
    }
    if let Some(v) = unified.params.radiation_moon_sv_month {
        params.radiation_moon_sv_month = v;
    }
    if let Some(v) = unified.params.radiation_career_limit_sv {
        params.radiation_career_limit_sv = v;
    }
    if let Some(v) = unified.params.spoilage_food {
        params.spoilage_food = v;
    }
    if let Some(v) = unified.params.spoilage_energy {
        params.spoilage_energy = v;
    }
    if let Some(v) = unified.params.spoilage_water {
        params.spoilage_water = v;
    }
}

/// Create a `SimulationConfig` from a unified TOML string.
///
/// Parses the TOML, creates default configs, then applies overrides.
pub fn from_unified_toml(
    toml_str: &str,
) -> Result<(SimulationConfig, PolicyConfig, SimulationParams), String> {
    let unified = UnifiedConfig::from_toml(toml_str)
        .map_err(|e| format!("Failed to parse unified config: {e}"))?;

    let mut sim_config = SimulationConfig::default_150_year();
    let mut policy = PolicyConfig::default();
    let mut params = SimulationParams::default();

    apply_unified_config(&unified, &mut sim_config, &mut policy, &mut params);
    sim_config.policy = policy.clone();

    Ok((sim_config, policy, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_unified_config_produces_defaults() {
        let (sim, policy, params) = from_unified_toml("").unwrap();
        assert_eq!(sim.seed, 42);
        assert!(policy.disasters_enabled);
        assert!(params.stress_baseline > 0.0);
    }

    #[test]
    fn test_override_specific_fields() {
        let toml = r#"
[simulation]
years = 500
seed = 999

[policy]
disasters_enabled = false
trade_openness = 0.3

[params]
stress_baseline = 0.01
"#;
        let (sim, policy, params) = from_unified_toml(toml).unwrap();
        assert_eq!(sim.seed, 999);
        assert_eq!(sim.total_ticks, 500 * 12);
        assert!(!policy.disasters_enabled);
        assert!((policy.trade_openness - 0.3).abs() < 0.001);
        assert!((params.stress_baseline - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_parse_shipped_default_config() {
        let toml_str = include_str!("../scenarios/unified_default.toml");
        let (sim, policy, params) = from_unified_toml(toml_str).unwrap();
        assert_eq!(sim.seed, 42);
        assert_eq!(sim.total_ticks, 300 * 12);
        assert!(policy.trust_weighted_governance);
        assert!(policy.hybrid_earth);
        assert!((params.stress_baseline - 0.004).abs() < 0.001);
        assert!((params.radiation_mars_sv_month - 0.020).abs() < 0.001);
    }

    #[test]
    fn test_feedback_loops_available() {
        let unified = UnifiedConfig::from_toml(
            r#"
[feedback_loops]
disaster_trauma = true
harmony_policy = false
"#,
        )
        .unwrap();
        assert!(unified.feedback_loops.disaster_trauma);
        assert!(!unified.feedback_loops.harmony_policy);
    }
}
