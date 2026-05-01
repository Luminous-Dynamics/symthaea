// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Civilization observables: derived metrics for long-horizon analysis.
//!
//! These functions compute second-order properties of the simulation state
//! (inequality, stagnation, divergence, trauma) that drive epoch evaluation
//! and faction dynamics.

use crate::epoch::EpochSnapshot;
use crate::governance::WorldGovernance;
use crate::world::World;

/// Classification of the phi (consciousness) trend over a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiTrend {
    Rising,
    Oscillating,
    Plateau,
    Declining,
}

impl std::fmt::Display for PhiTrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhiTrend::Rising => write!(f, "Rising"),
            PhiTrend::Oscillating => write!(f, "Oscillating"),
            PhiTrend::Plateau => write!(f, "Plateau"),
            PhiTrend::Declining => write!(f, "Declining"),
        }
    }
}

/// Classify the phi trend over a sliding window using linear regression.
///
/// - Oscillating: >4 sign changes in the first-differences within the window.
/// - Plateau: |slope| < 0.0001.
/// - Rising/Declining: by sign of slope.
pub fn classify_phi_trend(history: &[f64], window: usize) -> PhiTrend {
    if history.len() < 2 {
        return PhiTrend::Plateau;
    }

    let start = if history.len() > window {
        history.len() - window
    } else {
        0
    };
    let data = &history[start..];
    let n = data.len();
    if n < 2 {
        return PhiTrend::Plateau;
    }

    // Check for oscillation: count sign changes in first differences
    let mut sign_changes = 0u32;
    let mut prev_diff = 0.0f64;
    for i in 1..n {
        let diff = data[i] - data[i - 1];
        if i > 1 && diff * prev_diff < 0.0 {
            sign_changes += 1;
        }
        prev_diff = diff;
    }
    if sign_changes > 4 {
        return PhiTrend::Oscillating;
    }

    // Linear regression: slope = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - sum(x)^2)
    let n_f = n as f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut sum_x2 = 0.0f64;
    for (i, &val) in data.iter().enumerate() {
        let x = i as f64;
        sum_x += x;
        sum_y += val;
        sum_xy += x * val;
        sum_x2 += x * x;
    }
    let denom = n_f * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return PhiTrend::Plateau;
    }
    let slope = (n_f * sum_xy - sum_x * sum_y) / denom;

    if slope.abs() < 0.0001 {
        PhiTrend::Plateau
    } else if slope > 0.0 {
        PhiTrend::Rising
    } else {
        PhiTrend::Declining
    }
}

/// Fraction of Guardian-tier agents (phi >= 0.8) whose parents were also Guardian-tier.
///
/// Returns 0.0 if no Guardians exist or no parent data is available.
pub fn elite_persistence_index(world: &World, _current_tick: u32) -> f64 {
    let guardians: Vec<_> = world
        .agents
        .iter()
        .filter(|a| a.is_alive() && a.consciousness.phi() >= 0.8)
        .collect();

    if guardians.is_empty() {
        return 0.0;
    }

    let mut inherited = 0usize;
    let mut with_parents = 0usize;

    for g in &guardians {
        if let Some((mother_id, father_id)) = g.parent_ids {
            with_parents += 1;
            let parent_guardian = world
                .agents
                .iter()
                .any(|a| (a.id == mother_id || a.id == father_id) && a.consciousness.phi() >= 0.8);
            if parent_guardian {
                inherited += 1;
            }
        }
    }

    if with_parents == 0 {
        return 0.0;
    }

    inherited as f64 / with_parents as f64
}

/// Count amendments in the governance amendment_history within the last `window_ticks`.
pub fn amendment_rate(governance: &WorldGovernance, current_tick: u32, window_ticks: u32) -> f64 {
    let cutoff = current_tick.saturating_sub(window_ticks);
    governance
        .amendment_history
        .iter()
        .filter(|(tick, _)| *tick >= cutoff)
        .count() as f64
}

/// Innovation stagnation index based on recent technology growth.
///
/// Examines the last `window` entries (default 120) of the tech history.
/// If max growth over that window < 0.01, stagnation = 1.0 - growth/0.01.
pub fn innovation_stagnation_index(
    tech_history: &[(u32, f64)],
    _current_tick: u32,
    window: usize,
) -> f64 {
    if tech_history.len() < 2 {
        return 0.0;
    }

    let start = if tech_history.len() > window {
        tech_history.len() - window
    } else {
        0
    };
    let recent = &tech_history[start..];

    let min_tech = recent.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
    let max_tech = recent
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);

    let growth = max_tech - min_tech;
    if growth >= 0.01 {
        0.0
    } else {
        1.0 - growth / 0.01
    }
}

/// Mean pairwise cultural distance between worlds.
pub fn inter_world_divergence(worlds: &[World]) -> f64 {
    if worlds.len() < 2 {
        return 0.0;
    }

    let mut total_distance = 0.0;
    let mut pair_count = 0usize;

    for i in 0..worlds.len() {
        for j in (i + 1)..worlds.len() {
            total_distance += worlds[i].culture.cultural_distance(&worlds[j].culture);
            pair_count += 1;
        }
    }

    if pair_count == 0 {
        return 0.0;
    }

    total_distance / pair_count as f64
}

/// Gini coefficient of consciousness (phi) values across agents.
///
/// Uses the sorted-list algorithm:
/// G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
pub fn consciousness_gini(phi_values: &[f64]) -> f64 {
    if phi_values.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = phi_values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len() as f64;
    let sum_val: f64 = sorted.iter().sum();
    if sum_val <= 0.0 {
        return 0.0;
    }

    let weighted_sum: f64 = sorted
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64 + 1.0) * v)
        .sum();

    let gini = (2.0 * weighted_sum) / (n * sum_val) - (n + 1.0) / n;
    gini.clamp(0.0, 1.0)
}

/// Mean pairwise genetic distance (speciation index) between worlds.
///
/// Based on allele frequency divergence: worlds with different age distributions
/// and generation depths diverge genetically.
pub fn speciation_index(worlds: &[World], _current_tick: u32) -> f64 {
    if worlds.len() < 2 {
        return 0.0;
    }

    let mut total_dist = 0.0;
    let mut pairs = 0usize;

    for i in 0..worlds.len() {
        for j in (i + 1)..worlds.len() {
            let gen_a = mean_generation(&worlds[i]);
            let gen_b = mean_generation(&worlds[j]);
            // Generation divergence: absolute difference scaled by max
            let max_gen = gen_a.max(gen_b).max(1.0);
            let gen_dist = (gen_a - gen_b).abs() / max_gen;

            // Cultural distance as proxy for reproductive isolation
            let cult_dist = worlds[i].culture.cultural_distance(&worlds[j].culture);

            total_dist += gen_dist * 0.6 + cult_dist * 0.4;
            pairs += 1;
        }
    }

    if pairs == 0 {
        return 0.0;
    }

    (total_dist / pairs as f64).clamp(0.0, 1.0)
}

/// Mean generation number of living agents in a world.
fn mean_generation(world: &World) -> f64 {
    let living: Vec<f64> = world
        .agents
        .iter()
        .filter(|a| a.is_alive())
        .map(|a| a.generation as f64)
        .collect();
    if living.is_empty() {
        return 0.0;
    }
    living.iter().sum::<f64>() / living.len() as f64
}

/// Mean trauma level across living agents in a world.
pub fn trauma_level(world: &World) -> f64 {
    let living: Vec<f64> = world
        .agents
        .iter()
        .filter(|a| a.is_alive())
        .map(|a| a.trauma_level)
        .collect();
    if living.is_empty() {
        return 0.0;
    }
    living.iter().sum::<f64>() / living.len() as f64
}

/// Count CVS drops > 0.1 that were followed by recovery within 120 ticks.
pub fn recovery_count(snapshots: &[EpochSnapshot]) -> usize {
    if snapshots.len() < 2 {
        return 0;
    }

    let mut count = 0usize;

    for i in 1..snapshots.len() {
        let drop = snapshots[i - 1].civilization_viability_score
            - snapshots[i].civilization_viability_score;
        if drop > 0.1 {
            // Look for recovery within 120 ticks
            let drop_tick = snapshots[i].tick;
            let recovered = snapshots[i..].iter().any(|s| {
                s.tick <= drop_tick + 120
                    && s.civilization_viability_score
                        >= snapshots[i - 1].civilization_viability_score - 0.02
            });
            if recovered {
                count += 1;
            }
        }
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::world::{CulturalProfile, World, WorldResources};

    fn make_test_world_obs(pop: usize) -> World {
        let mut world = World {
            id: 0,
            name: "Test".into(),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents: Vec::new(),
            next_agent_id: 0,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 100_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: crate::knowledge::WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony: crate::harmony::HarmonyTracker::new(),
            governance: crate::governance::WorldGovernance::new(),
            metabolism_state: crate::metabolism::MetabolismState::default(),
            currency_state: crate::currency::WorldCurrencyState::default(),
            policy_state: crate::proposals::PolicyState::default(),
            power_generation_kw: 0.0,
            power_demand_kw: 0.0,
            narrative_identity: crate::world::NarrativeIdentity::default(),
            maintenance_hours_required: 0.0,
            maintenance_hours_available: 0.0,
            bus_factor_critical: 0,
            pathogen_pressure: 0.0,
            civilizational_phi: 0.0,
            trust_level: 0.7,
            earth_funding: 1.0,
            mortality_alpha_mult: 1.0,
            mortality_beta_mult: 1.0,
            mortality_lambda_mult: 1.0,
            reproduction_viable: true,
            ecosystem_balance: 1.0,
            fertility_multiplier: 1.0,
            automation_level: 0.0,
            explorations_completed: 0,
            project_manager: crate::projects::ProjectManager::new(),
            habitat: crate::habitat::HabitatComplex::default(),
            fleet: crate::robotics::RoboticFleet::default(),
            diplomatic_relations: std::collections::HashMap::new(),
            zones: Vec::new(),
            moral_memories: Vec::new(),
            institutional_ethics: crate::agent::EthicalOrientation::default(),
        };
        for i in 0..pop {
            let agent = CivAgent {
                id: i as u64,
                birth_tick: 0,
                death_tick: None,
                sex: if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                },
                world_id: 0,
                health: 0.9,
                skills: SkillVector::new(),
                education_level: 0.3,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: crate::needs::PsychologicalNeeds::new(),
                tend_balance: 0.0,
                parent_ids: None,
                faction_id: None,
                generation: 0,
                trauma_level: 0.0,
                cumulative_dose_sv: 0.0,
                adversarial: None,
                coordination_understanding: 0.0,
                mycel_score: 0.1,
                sap_balance: 100.0,
                is_biological: true,
                wounds: Vec::new(),
                ethics: crate::agent::EthicalOrientation::default(),
                sovereign_profile: crate::sovereign_profile::SovereignProfile::zero(),
                justice: crate::sub_passport::RestorativeJustice::new(),
            };
            world.agents.push(agent);
            world.next_agent_id += 1;
        }
        world
    }

    #[test]
    fn test_phi_trend_rising() {
        let history: Vec<f64> = (0..20).map(|i| 0.1 + i as f64 * 0.01).collect();
        assert_eq!(classify_phi_trend(&history, 20), PhiTrend::Rising);
    }

    #[test]
    fn test_phi_trend_declining() {
        let history: Vec<f64> = (0..20).map(|i| 0.5 - i as f64 * 0.01).collect();
        assert_eq!(classify_phi_trend(&history, 20), PhiTrend::Declining);
    }

    #[test]
    fn test_phi_trend_plateau() {
        let history: Vec<f64> = (0..20).map(|_| 0.5).collect();
        assert_eq!(classify_phi_trend(&history, 20), PhiTrend::Plateau);
    }

    #[test]
    fn test_phi_trend_oscillating() {
        // Create oscillating pattern with >4 sign changes
        let history: Vec<f64> = (0..20)
            .map(|i| 0.5 + if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();
        assert_eq!(classify_phi_trend(&history, 20), PhiTrend::Oscillating);
    }

    #[test]
    fn test_consciousness_gini_uniform() {
        // All equal phi -> Gini = 0
        let phis: Vec<f64> = vec![0.5; 100];
        let g = consciousness_gini(&phis);
        assert!(g < 0.01, "Uniform phi should have Gini ~0, got {g}");
    }

    #[test]
    fn test_consciousness_gini_inequality() {
        // One agent has all consciousness
        let mut phis: Vec<f64> = vec![0.0; 99];
        phis.push(1.0);
        let g = consciousness_gini(&phis);
        assert!(g > 0.9, "Extreme inequality should have Gini >0.9, got {g}");
    }

    #[test]
    fn test_inter_world_divergence_single() {
        let world = make_test_world_obs(10);
        assert_eq!(inter_world_divergence(&[world]), 0.0);
    }

    #[test]
    fn test_inter_world_divergence_two() {
        let w1 = make_test_world_obs(10);
        let mut w2 = make_test_world_obs(10);
        // Modify culture of w2 to create distance
        w2.culture.harmony_weights = [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05];
        let d = inter_world_divergence(&[w1, w2]);
        assert!(
            d > 0.0,
            "Different cultures should have divergence > 0, got {d}"
        );
    }

    #[test]
    fn test_trauma_level_zero() {
        let world = make_test_world_obs(10);
        assert_eq!(trauma_level(&world), 0.0);
    }

    #[test]
    fn test_recovery_count_empty() {
        assert_eq!(recovery_count(&[]), 0);
    }

    #[test]
    fn test_innovation_stagnation_flat() {
        let history: Vec<(u32, f64)> = (0..200).map(|i| (i, 5.0)).collect();
        let idx = innovation_stagnation_index(&history, 200, 120);
        assert!(
            (idx - 1.0).abs() < 0.01,
            "Flat tech should be stagnant, got {idx}"
        );
    }

    #[test]
    fn test_innovation_stagnation_growing() {
        let history: Vec<(u32, f64)> = (0..200).map(|i| (i, 1.0 + i as f64 * 0.01)).collect();
        let idx = innovation_stagnation_index(&history, 200, 120);
        assert_eq!(idx, 0.0, "Growing tech should not be stagnant");
    }

    #[test]
    fn test_elite_persistence_no_guardians() {
        let world = make_test_world_obs(10);
        assert_eq!(elite_persistence_index(&world, 100), 0.0);
    }

    #[test]
    fn test_speciation_single_world() {
        let world = make_test_world_obs(10);
        assert_eq!(speciation_index(&[world], 100), 0.0);
    }
}
