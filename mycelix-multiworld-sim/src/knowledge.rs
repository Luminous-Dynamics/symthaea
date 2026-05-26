// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Romer endogenous growth model with cross-world knowledge diffusion.
//!
//! Innovation follows Romer (1990): non-rival knowledge accumulates endogenously,
//! with technology levels growing as a function of researcher count, existing
//! knowledge stock, and mean education. Breakthroughs (rare, high-impact events)
//! punctuate the gradual improvement.
//!
//! Cross-world diffusion allows colonies to absorb technology from more advanced
//! worlds, modulated by cultural distance and contact frequency.

use mycelix_desci_core::LEMCube;
use serde::{Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;

use crate::agent::CivAgent;
use crate::events::{CivEvent, CivEventType};
use crate::stochastic::StochasticEngine;
use crate::world::World;

/// Number of technology sectors (matches economy sectors).
pub const NUM_SECTORS: usize = 8;

/// Base innovation rate per tick (before scaling by researchers/tech/education).
/// Space colonies are research-focused by necessity — innovation rate is higher
/// than terrestrial baseline because every problem is novel.
const BASE_INNOVATION_RATE: f64 = 0.05;

/// Minimum tech level increment per innovation event.
const MIN_TECH_INCREMENT: f64 = 0.01;

/// Maximum tech level increment per innovation event.
const MAX_TECH_INCREMENT: f64 = 0.05;

/// Breakthrough probability per tick.
const BREAKTHROUGH_PROB: f64 = 0.001;

/// Minimum breakthrough magnitude.
const BREAKTHROUGH_MIN: f64 = 0.1;

/// Maximum breakthrough magnitude.
const BREAKTHROUGH_MAX: f64 = 0.5;

/// Diffusion absorption coefficient.
const DIFFUSION_COEFFICIENT: f64 = 0.1;

/// Engineering sector index (target for LCF breakthroughs).
const ENERGY_SECTOR: usize = 0;

/// Sector names for breakthrough descriptions.
const SECTOR_NAMES: [&str; NUM_SECTORS] = [
    "engineering",
    "agriculture",
    "medicine",
    "governance",
    "science",
    "education",
    "art & culture",
    "logistics",
];

/// A major technological breakthrough event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Breakthrough {
    /// Unique identifier for this breakthrough claim.
    pub id: Uuid,
    /// Tick when the breakthrough occurred.
    pub tick: u32,
    /// Sector index (0-7).
    pub sector: usize,
    /// How much the technology level jumped.
    pub magnitude: f64,
    /// Human-readable description.
    pub description: String,
    /// Epistemic classification of the breakthrough.
    pub lem: LEMCube,
}

/// World knowledge system tracking technology levels and innovation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldKnowledge {
    /// Per-sector technology levels (start at 1.0).
    pub technology_levels: [f64; NUM_SECTORS],
    /// Total accumulated knowledge claims.
    pub total_claims: u32,
    /// Number of agents actively doing research this tick.
    pub active_researchers: usize,
    /// Current innovation rate (innovations per tick).
    pub innovation_rate: f64,
    /// Specialization depth: 0.0 = generalist, 1.0 = deeply specialized.
    pub specialization_depth: f64,
    /// Generalist breadth: inverse of specialization.
    pub generalist_breadth: f64,
    /// Knowledge received from other worlds this tick.
    pub diffusion_received: f64,
    /// History of major breakthroughs.
    pub breakthroughs: Vec<Breakthrough>,
    /// LCF breakthrough probability per tick (configurable, active in Epochs 3-4).
    pub lcf_probability: f64,
    /// Sampled tech level history: (tick, mean_tech_level), sampled every 12 ticks.
    pub tech_history: Vec<(u32, f64)>,
    /// Whether stagnation has been detected.
    pub stagnation_detected: bool,
    /// Tick at which stagnation was first detected.
    pub stagnation_start_tick: Option<u32>,

    /// Epistemic membrane health: 0.0 (total hallucination) to 1.0 (perfect truth).
    /// Gated by STARK verification and Biome Tensor consistency.
    pub epistemic_membrane: f64,

    /// Algebraic connectivity (λ2) of the researcher network.
    /// From phi-lab: 0.0 = disconnected, 1.0 = hyper-coherent.
    pub network_lambda_2: f64,

    /// Number of breakthroughs permanently anchored in Resontia vaults.
    pub resontia_anchored_claims: u32,

    /// Critical system dependency tracking.
    #[serde(default)]
    pub critical_system_coverage: CriticalSystemCoverage,
}

/// Tracks which critical colony systems have adequate skilled operators.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CriticalSystemCoverage {
    pub eclss_operators: u32,
    pub power_operators: u32,
    pub agriculture_operators: u32,
    pub medical_operators: u32,
    pub comms_operators: u32,
    pub fabrication_operators: u32,
    pub education_operators: u32,
    pub governance_operators: u32,
    pub systems_at_risk: u32,
    pub systems_failing: u32,
}

impl CriticalSystemCoverage {
    pub fn compute(agents: &[crate::agent::CivAgent], current_tick: u32) -> Self {
        let living: Vec<_> = agents
            .iter()
            .filter(|a| a.is_alive() && a.life_stage(current_tick).can_work())
            .collect();

        let count_skilled = |sector: usize, min_skill: f64| -> u32 {
            living
                .iter()
                .filter(|a| a.skills.as_slice()[sector] > min_skill)
                .count() as u32
        };

        let eclss = count_skilled(0, 0.3);
        let power = count_skilled(0, 0.4).min(count_skilled(4, 0.3));
        let agriculture = count_skilled(1, 0.3);
        let medical = count_skilled(2, 0.3);
        let comms = count_skilled(0, 0.3);
        let fabrication = count_skilled(0, 0.4).min(count_skilled(7, 0.2));
        let education = count_skilled(5, 0.3);
        let governance = count_skilled(3, 0.3);

        let systems = [
            eclss,
            power,
            agriculture,
            medical,
            comms,
            fabrication,
            education,
            governance,
        ];
        let at_risk = systems.iter().filter(|&&s| s <= 1).count() as u32;
        let failing = systems.iter().filter(|&&s| s == 0).count() as u32;

        Self {
            eclss_operators: eclss,
            power_operators: power,
            agriculture_operators: agriculture,
            medical_operators: medical,
            comms_operators: comms,
            fabrication_operators: fabrication,
            education_operators: education,
            governance_operators: governance,
            systems_at_risk: at_risk,
            systems_failing: failing,
        }
    }
}

pub fn compute_world_tech_levels(agents: &[crate::agent::CivAgent]) -> [f64; NUM_SECTORS] {
    let mut levels = [1.0; NUM_SECTORS];
    for sector in 0..NUM_SECTORS {
        let mut skills: Vec<f64> = agents
            .iter()
            .filter(|a| a.is_alive())
            .map(|a| a.skills.as_slice()[sector])
            .collect();

        if skills.is_empty() {
            continue;
        }
        skills.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top_n = (skills.len() / 10).max(1);
        let top_mean: f64 = skills[..top_n].iter().sum::<f64>() / top_n as f64;
        levels[sector] = 1.0 + top_mean * 4.0;
    }
    levels
}

impl WorldKnowledge {
    pub fn new() -> Self {
        Self {
            technology_levels: [1.0; NUM_SECTORS],
            total_claims: 0,
            active_researchers: 0,
            innovation_rate: 0.0,
            specialization_depth: 0.0,
            generalist_breadth: 1.0,
            diffusion_received: 0.0,
            breakthroughs: Vec::new(),
            lcf_probability: 0.0,
            tech_history: Vec::new(),
            stagnation_detected: false,
            stagnation_start_tick: None,
            epistemic_membrane: 1.0,
            network_lambda_2: 0.5,
            resontia_anchored_claims: 0,
            critical_system_coverage: CriticalSystemCoverage::default(),
        }
    }

    pub fn tick_knowledge(
        &mut self,
        world: &World,
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> Vec<CivEvent> {
        self.tick_knowledge_standalone(world.id, &world.name, &world.agents, current_tick, rng)
    }

    pub fn tick_knowledge_standalone(
        &mut self,
        world_id: u32,
        world_name: &str,
        agents: &[CivAgent],
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();
        let population = agents.len();
        if population == 0 {
            return events;
        }

        let researcher_equivalent: f64 = agents
            .iter()
            .filter(|a| a.is_alive())
            .map(|a| {
                if a.skills.strongest() == "science" {
                    1.0
                } else {
                    0.2
                }
            })
            .sum();
        self.active_researchers = researcher_equivalent.ceil() as usize;

        let mean_education = {
            let living: Vec<f64> = agents
                .iter()
                .filter(|a| a.is_alive())
                .map(|a| a.education_level)
                .collect();
            if living.is_empty() {
                0.0
            } else {
                living.iter().sum::<f64>() / living.len() as f64
            }
        };

        let tech_sum: f64 = self.technology_levels.iter().sum();
        self.innovation_rate = BASE_INNOVATION_RATE
            * (self.active_researchers as f64).sqrt()
            * tech_sum.powf(0.3)
            * (mean_education + 0.01).powf(0.2);

        // --- PHASE 17: TOPOLOGICAL EPISTEMIC INTEGRITY ---
        let required_quorum = (population as f64 * 0.05).max(10.0);
        let quorum_fraction = (self.active_researchers as f64 / required_quorum).min(1.0);
        let coherence_factor = (self.network_lambda_2 / 0.4).min(1.0);
        let target_membrane = quorum_fraction * coherence_factor;
        self.epistemic_membrane = (self.epistemic_membrane * 0.95) + (target_membrane * 0.05);

        let monthly_prob = (self.innovation_rate / 12.0).clamp(0.0, 0.95);
        if rng.bernoulli(monthly_prob) {
            let sector = (rng.next_u64() % NUM_SECTORS as u64) as usize;
            let increment =
                MIN_TECH_INCREMENT + rng.next_f64() * (MAX_TECH_INCREMENT - MIN_TECH_INCREMENT);
            let verified_increment = increment * self.epistemic_membrane;
            self.technology_levels[sector] += verified_increment;
            self.total_claims += 1;
            events.push(CivEvent::new(
                current_tick,
                Some(world_id),
                CivEventType::InnovationBreakthrough,
                format!(
                    "{}: innovation in {} (+{:.3} verified)",
                    world_name, SECTOR_NAMES[sector], verified_increment
                ),
            ));
        }

        if rng.bernoulli(BREAKTHROUGH_PROB) {
            let sector = (rng.next_u64() % NUM_SECTORS as u64) as usize;
            let magnitude =
                BREAKTHROUGH_MIN + rng.next_f64() * (BREAKTHROUGH_MAX - BREAKTHROUGH_MIN);
            let verified_magnitude = magnitude * self.epistemic_membrane;
            self.technology_levels[sector] += verified_magnitude;
            self.total_claims += 1;
            let bt = Breakthrough {
                id: Uuid::new_v4(),
                tick: current_tick,
                sector,
                magnitude: verified_magnitude,
                description: format!(
                    "Major breakthrough in {} (+{:.3} verified)",
                    SECTOR_NAMES[sector], verified_magnitude
                ),
                lem: LEMCube::new(
                    mycelix_desci_core::EmpiricalAxis::E3CryptographicallyProven,
                    mycelix_desci_core::NormativeAxis::N2Network,
                    mycelix_desci_core::MaterialityAxis::M2Persistent,
                ),
            };
            self.breakthroughs.push(bt.clone());
            if bt.lem.empirical == mycelix_desci_core::EmpiricalAxis::E4PubliclyReproducible
                && bt.lem.materiality == mycelix_desci_core::MaterialityAxis::M3Foundational
            {
                self.resontia_anchored_claims += 1;
                info!(
                    "💎 [Resontia] Breakthrough {} anchored in subterranean vaults.",
                    bt.id
                );
            }
            events.push(CivEvent::new(
                current_tick,
                Some(world_id),
                CivEventType::InnovationBreakthrough,
                format!(
                    "{}: BREAKTHROUGH in {} (+{:.3} verified)",
                    world_name, SECTOR_NAMES[sector], verified_magnitude
                ),
            ));
        }

        if self.lcf_probability > 0.0 && rng.bernoulli(self.lcf_probability) {
            let magnitude = 1.0 + rng.next_f64() * 2.0;
            self.technology_levels[ENERGY_SECTOR] += magnitude;
            self.total_claims += 1;
            let bt = Breakthrough {
                id: Uuid::new_v4(),
                tick: current_tick,
                sector: ENERGY_SECTOR,
                magnitude,
                description: format!(
                    "LCF energy breakthrough! Engineering tech jumps by {magnitude:.2}"
                ),
                lem: LEMCube::new(
                    mycelix_desci_core::EmpiricalAxis::E4PubliclyReproducible,
                    mycelix_desci_core::NormativeAxis::N3Axiomatic,
                    mycelix_desci_core::MaterialityAxis::M3Foundational,
                ),
            };
            self.breakthroughs.push(bt.clone());
            if bt.lem.empirical == mycelix_desci_core::EmpiricalAxis::E4PubliclyReproducible
                && bt.lem.materiality == mycelix_desci_core::MaterialityAxis::M3Foundational
            {
                self.resontia_anchored_claims += 1;
                info!(
                    "💎 [Resontia] LCF Breakthrough {} anchored in subterranean vaults.",
                    bt.id
                );
            }
            events.push(CivEvent::new(
                current_tick,
                Some(world_id),
                CivEventType::InnovationBreakthrough,
                format!(
                    "{}: LCF ENERGY BREAKTHROUGH (+{:.2})",
                    world_name, magnitude
                ),
            ));
        }

        self.specialization_vs_breadth(population);
        if current_tick % 12 == 0 {
            self.tech_history
                .push((current_tick, self.mean_tech_level()));
        }
        self.detect_stagnation(current_tick);
        self.diffusion_received = 0.0;
        events
    }

    pub fn detect_stagnation(&mut self, tick: u32) {
        if self.tech_history.len() < 2 {
            return;
        }
        let window = 120.min(self.tech_history.len());
        let recent = &self.tech_history[self.tech_history.len() - window..];
        let growth = recent.last().map(|(_, v)| *v).unwrap_or(0.0)
            - recent.first().map(|(_, v)| *v).unwrap_or(0.0);
        if growth < 0.005 {
            if !self.stagnation_detected {
                self.stagnation_detected = true;
                self.stagnation_start_tick = Some(tick);
            }
        } else {
            self.stagnation_detected = false;
            self.stagnation_start_tick = None;
        }
    }

    pub fn technological_lock_in_index(&self) -> f64 {
        if self.generalist_breadth <= 0.01 {
            return 10.0;
        }
        self.specialization_depth / self.generalist_breadth
    }

    pub fn research_network_effect(num_connected_worlds: usize) -> f64 {
        ((num_connected_worlds as f64).sqrt() / 3.0).max(1.0)
    }

    pub fn receive_diffusion(
        &mut self,
        other: &WorldKnowledge,
        cultural_distance: f64,
        contact_frequency: f64,
    ) {
        let absorption = contact_frequency * (1.0 - cultural_distance) * DIFFUSION_COEFFICIENT;
        for i in 0..NUM_SECTORS {
            let gap = other.technology_levels[i] - self.technology_levels[i];
            if gap > 0.0 {
                let transfer = gap * absorption;
                self.technology_levels[i] += transfer;
                self.diffusion_received += transfer;
            }
        }
    }

    pub fn specialization_vs_breadth(&mut self, population: usize) {
        if population == 0 {
            self.specialization_depth = 0.0;
            self.generalist_breadth = 1.0;
            return;
        }
        let max_spec = 1.0 - (500.0 / population as f64).clamp(0.0, 1.0);
        let mean_tech = self.mean_tech_level();
        let variance: f64 = self
            .technology_levels
            .iter()
            .map(|&t| (t - mean_tech).powi(2))
            .sum::<f64>()
            / NUM_SECTORS as f64;
        let normalized_variance = (variance / (mean_tech * mean_tech + 0.001)).min(1.0);
        self.specialization_depth = normalized_variance.min(max_spec);
        self.generalist_breadth = 1.0 - self.specialization_depth;
    }

    pub fn mean_tech_level(&self) -> f64 {
        self.technology_levels.iter().sum::<f64>() / NUM_SECTORS as f64
    }

    pub fn sync_from_agents(&mut self, agents: &[crate::agent::CivAgent]) {
        let new_levels = compute_world_tech_levels(agents);
        for i in 0..NUM_SECTORS {
            self.technology_levels[i] = 0.7 * new_levels[i] + 0.3 * self.technology_levels[i];
        }
    }

    pub fn max_tech_sector(&self) -> (usize, f64) {
        self.technology_levels
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i, v))
            .unwrap_or((0, 1.0))
    }
}

impl Default for WorldKnowledge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
    use crate::stochastic::StochasticEngine;
    use crate::world::{ColonyParams, CulturalProfile, World, WorldResources};

    const TEST_TICK: u32 = 1000;

    fn make_world_with_researchers(n_researchers: usize, n_others: usize) -> World {
        let params = ColonyParams {
            id: 0,
            name: "TestWorld".into(),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 100_000.0,
        };
        let mut world = World::new_colony(params);
        let mut id = 0u64;

        for _ in 0..n_researchers {
            let mut skills = SkillVector::new();
            skills.learn(4, 0.8);
            world.agents.push(CivAgent {
                id,
                birth_tick: TEST_TICK - 30 * 12,
                death_tick: None,
                sex: BiologicalSex::Male,
                world_id: 0,
                health: 1.0,
                skills,
                education_level: 0.7,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: Default::default(),
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
                ethics: Default::default(),
                sovereign_profile: Default::default(),
                justice: Default::default(),
            });
            id += 1;
        }

        for _ in 0..n_others {
            let mut skills = SkillVector::new();
            skills.learn(1, 0.6);
            world.agents.push(CivAgent {
                id,
                birth_tick: TEST_TICK - 30 * 12,
                death_tick: None,
                sex: BiologicalSex::Female,
                world_id: 0,
                health: 1.0,
                skills,
                education_level: 0.3,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: Default::default(),
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
                ethics: Default::default(),
                sovereign_profile: Default::default(),
                justice: Default::default(),
            });
            id += 1;
        }
        world.next_agent_id = id;
        world
    }

    #[test]
    fn test_innovation_rate_scales_with_researchers() {
        let mut k1 = WorldKnowledge::new();
        let mut rng1 = StochasticEngine::new(42);
        let world_small = make_world_with_researchers(2, 10);
        k1.tick_knowledge(&world_small, TEST_TICK, &mut rng1);
        let rate_small = k1.innovation_rate;

        let mut k2 = WorldKnowledge::new();
        let mut rng2 = StochasticEngine::new(42);
        let world_large = make_world_with_researchers(20, 80);
        k2.tick_knowledge(&world_large, TEST_TICK, &mut rng2);
        let rate_large = k2.innovation_rate;

        assert!(rate_large > rate_small);
    }

    #[test]
    fn test_breakthrough_logged() {
        let mut knowledge = WorldKnowledge::new();
        knowledge.breakthroughs.push(Breakthrough {
            id: Uuid::new_v4(),
            tick: 100,
            sector: 4,
            magnitude: 0.3,
            description: "Test breakthrough".into(),
            lem: LEMCube::new(
                mycelix_desci_core::EmpiricalAxis::E0Unverified,
                mycelix_desci_core::NormativeAxis::N0Personal,
                mycelix_desci_core::MaterialityAxis::M0None,
            ),
        });
        assert_eq!(knowledge.breakthroughs.len(), 1);
    }
}
