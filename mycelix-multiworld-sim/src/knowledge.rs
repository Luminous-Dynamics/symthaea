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

use serde::{Deserialize, Serialize};

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
    /// Tick when the breakthrough occurred.
    pub tick: u32,
    /// Sector index (0-7).
    pub sector: usize,
    /// How much the technology level jumped.
    pub magnitude: f64,
    /// Human-readable description.
    pub description: String,
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
}

impl WorldKnowledge {
    /// Create a new knowledge system at baseline technology.
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
        }
    }

    /// Run one tick of the knowledge system.
    ///
    /// Counts researchers (agents whose strongest skill is science), computes
    /// the Romer innovation rate, rolls for incremental innovations and rare
    /// breakthroughs, and optionally checks for LCF energy breakthroughs.
    pub fn tick_knowledge(
        &mut self,
        world: &World,
        current_tick: u32,
        rng: &mut StochasticEngine,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();
        let population = world.population();

        if population == 0 {
            return events;
        }

        // --- Count researchers (agents whose strongest skill is science, sector 4) ---
        // Count researchers: science-primary agents count as 1.0, all other working
        // adults count as 0.2 (everyone contributes to knowledge in a small colony).
        // This prevents the "zero researchers" problem in small populations.
        let researcher_equivalent: f64 = world
            .agents
            .iter()
            .filter(|a| a.is_alive() && a.life_stage(current_tick).can_work())
            .map(|a| if a.skills.strongest() == "science" { 1.0 } else { 0.2 })
            .sum();
        self.active_researchers = researcher_equivalent.ceil() as usize;

        // --- Mean education level ---
        let mean_education = {
            let living: Vec<f64> = world
                .agents
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

        // --- Compute innovation rate (Romer endogenous growth) ---
        // rate = base * sqrt(researchers) * (sum(tech))^0.3 * mean_education^0.2
        let tech_sum: f64 = self.technology_levels.iter().sum();
        self.innovation_rate = BASE_INNOVATION_RATE
            * (self.active_researchers as f64).sqrt()
            * tech_sum.powf(0.3)
            * (mean_education + 0.01).powf(0.2);

        // --- Roll for incremental innovation (monthly probability) ---
        let monthly_prob = (self.innovation_rate / 12.0).clamp(0.0, 0.95);
        if rng.bernoulli(monthly_prob) {
            let sector = (rng.next_u64() % NUM_SECTORS as u64) as usize;
            let increment =
                MIN_TECH_INCREMENT + rng.next_f64() * (MAX_TECH_INCREMENT - MIN_TECH_INCREMENT);
            self.technology_levels[sector] += increment;
            self.total_claims += 1;

            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::InnovationBreakthrough,
                format!(
                    "{}: innovation in {} (+{:.3}), tech level now {:.3}",
                    world.name,
                    SECTOR_NAMES[sector],
                    increment,
                    self.technology_levels[sector],
                ),
            ));
        }

        // --- Roll for breakthrough (rare, high-impact) ---
        if rng.bernoulli(BREAKTHROUGH_PROB) {
            let sector = (rng.next_u64() % NUM_SECTORS as u64) as usize;
            let magnitude =
                BREAKTHROUGH_MIN + rng.next_f64() * (BREAKTHROUGH_MAX - BREAKTHROUGH_MIN);
            self.technology_levels[sector] += magnitude;
            self.total_claims += 1;

            let bt = Breakthrough {
                tick: current_tick,
                sector,
                magnitude,
                description: format!(
                    "Major breakthrough in {} (+{:.3})",
                    SECTOR_NAMES[sector], magnitude
                ),
            };
            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::InnovationBreakthrough,
                format!(
                    "{}: BREAKTHROUGH in {} (+{:.3})! Tech level: {:.3}",
                    world.name,
                    SECTOR_NAMES[sector],
                    magnitude,
                    self.technology_levels[sector],
                ),
            ));
            self.breakthroughs.push(bt);
        }

        // --- LCF breakthrough (configurable, Epochs 3-4) ---
        if self.lcf_probability > 0.0 && rng.bernoulli(self.lcf_probability) {
            let magnitude = 1.0 + rng.next_f64() * 2.0; // dramatic: 1.0-3.0
            self.technology_levels[ENERGY_SECTOR] += magnitude;
            self.total_claims += 1;

            let bt = Breakthrough {
                tick: current_tick,
                sector: ENERGY_SECTOR,
                magnitude,
                description: format!(
                    "LCF energy breakthrough! Engineering tech jumps by {magnitude:.2}"
                ),
            };
            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::InnovationBreakthrough,
                format!(
                    "{}: LCF ENERGY BREAKTHROUGH (+{:.2})! Engineering tech: {:.3}",
                    world.name,
                    magnitude,
                    self.technology_levels[ENERGY_SECTOR],
                ),
            ));
            self.breakthroughs.push(bt);
        }

        // --- Update specialization vs breadth ---
        self.specialization_vs_breadth(population);

        // Reset diffusion counter for this tick
        self.diffusion_received = 0.0;

        events
    }

    /// Receive knowledge diffusion from another world.
    ///
    /// For each sector where the other world is more advanced, absorb a fraction
    /// of the technology gap:
    ///
    /// `diffusion = (other.tech - self.tech) * contact_frequency * (1 - cultural_distance) * 0.1`
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

    /// Update specialization depth based on population size.
    ///
    /// Small populations are forced to be generalists (everyone must cover
    /// multiple roles). Specialization capacity grows with population:
    ///
    /// `max_specialization = 1.0 - (500.0 / population).clamp(0.0, 1.0)`
    pub fn specialization_vs_breadth(&mut self, population: usize) {
        if population == 0 {
            self.specialization_depth = 0.0;
            self.generalist_breadth = 1.0;
            return;
        }

        let max_spec = 1.0 - (500.0 / population as f64).clamp(0.0, 1.0);

        // Actual specialization from tech variance
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

    /// Mean technology level across all sectors.
    pub fn mean_tech_level(&self) -> f64 {
        self.technology_levels.iter().sum::<f64>() / NUM_SECTORS as f64
    }

    /// Sector with the highest technology level. Returns (index, value).
    pub fn max_tech_sector(&self) -> (usize, f64) {
        self.technology_levels
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
    use crate::world::{CulturalProfile, World, WorldResources};

    /// Reference tick used by all knowledge tests. Agents are born relative to this.
    const TEST_TICK: u32 = 1000;

    fn make_world_with_researchers(n_researchers: usize, n_others: usize) -> World {
        let mut agents = Vec::new();
        let mut id = 0u64;

        // Researchers: science (sector 4) is their strongest skill
        for _ in 0..n_researchers {
            let mut skills = SkillVector::new();
            skills.learn(4, 0.8); // science
            agents.push(CivAgent {
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
                needs: crate::needs::PsychologicalNeeds::new(),
            tend_balance: 0.0,
            });
            id += 1;
        }

        // Others: agriculture (sector 1)
        for _ in 0..n_others {
            let mut skills = SkillVector::new();
            skills.learn(1, 0.6); // agriculture
            agents.push(CivAgent {
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
                needs: crate::needs::PsychologicalNeeds::new(),
            tend_balance: 0.0,
            });
            id += 1;
        }

        World {
            id: 0,
            name: "TestWorld".into(),
            location: "Moon".into(),
            founded_tick: 0,
            parent_world_id: None,
            agents,
            next_agent_id: id,
            resources: WorldResources::lunar_default(),
            culture: CulturalProfile::pioneer_default(),
            infrastructure_level: 0.5,
            max_population: 10_000,
            habitable_area_m2: 100_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony: crate::harmony::HarmonyTracker::new(),
        }
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

        assert!(
            rate_large > rate_small,
            "More researchers should mean higher rate: {rate_large} vs {rate_small}"
        );
    }

    #[test]
    fn test_diffusion_reduces_gap() {
        let mut receiver = WorldKnowledge::new();
        let mut sender = WorldKnowledge::new();
        sender.technology_levels = [3.0; NUM_SECTORS];

        let gap_before: f64 = (0..NUM_SECTORS)
            .map(|i| sender.technology_levels[i] - receiver.technology_levels[i])
            .sum();

        receiver.receive_diffusion(&sender, 0.2, 0.8);

        let gap_after: f64 = (0..NUM_SECTORS)
            .map(|i| sender.technology_levels[i] - receiver.technology_levels[i])
            .sum();

        assert!(
            gap_after < gap_before,
            "Diffusion should reduce gap: {gap_after} vs {gap_before}"
        );
        assert!(
            receiver.diffusion_received > 0.0,
            "Should record diffusion received"
        );
    }

    #[test]
    fn test_specialization_limited_by_population() {
        let mut knowledge = WorldKnowledge::new();
        // Create uneven tech levels to generate variance
        knowledge.technology_levels = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        knowledge.specialization_vs_breadth(50);
        let spec_small = knowledge.specialization_depth;

        knowledge.specialization_vs_breadth(5000);
        let spec_large = knowledge.specialization_depth;

        assert!(
            spec_large >= spec_small,
            "Large pop should allow more specialization: {spec_large} vs {spec_small}"
        );
        assert!(
            knowledge.generalist_breadth + knowledge.specialization_depth <= 1.001,
            "Breadth + depth should sum to ~1.0"
        );
    }

    #[test]
    fn test_breakthrough_logged() {
        let mut knowledge = WorldKnowledge::new();

        knowledge.breakthroughs.push(Breakthrough {
            tick: 100,
            sector: 4,
            magnitude: 0.3,
            description: "Test breakthrough".into(),
        });
        knowledge.technology_levels[4] += 0.3;

        assert_eq!(knowledge.breakthroughs.len(), 1);
        assert_eq!(knowledge.breakthroughs[0].sector, 4);
        assert!((knowledge.breakthroughs[0].magnitude - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_tech_levels_start_at_one() {
        let knowledge = WorldKnowledge::new();
        for &level in &knowledge.technology_levels {
            assert!(
                (level - 1.0).abs() < 0.001,
                "Tech should start at 1.0, got {level}"
            );
        }
        assert!((knowledge.mean_tech_level() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_lcf_breakthrough_jumps_energy_sector() {
        let mut knowledge = WorldKnowledge::new();
        knowledge.lcf_probability = 1.0; // guarantee trigger
        let world = make_world_with_researchers(5, 20);
        let mut rng = StochasticEngine::new(42);

        let energy_before = knowledge.technology_levels[ENERGY_SECTOR];
        let events = knowledge.tick_knowledge(&world, TEST_TICK, &mut rng);

        assert!(
            knowledge.technology_levels[ENERGY_SECTOR] > energy_before + 0.5,
            "LCF should dramatically jump energy sector: {} vs {energy_before}",
            knowledge.technology_levels[ENERGY_SECTOR]
        );

        let lcf_events: Vec<_> = events
            .iter()
            .filter(|e| e.description.contains("LCF"))
            .collect();
        assert!(
            !lcf_events.is_empty(),
            "Should generate an LCF breakthrough event"
        );
    }

    #[test]
    fn test_max_tech_sector() {
        let mut knowledge = WorldKnowledge::new();
        knowledge.technology_levels[3] = 5.0;
        let (idx, val) = knowledge.max_tech_sector();
        assert_eq!(idx, 3);
        assert!((val - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_empty_world_no_crash() {
        let mut knowledge = WorldKnowledge::new();
        let world = make_world_with_researchers(0, 0);
        let mut rng = StochasticEngine::new(42);

        let events = knowledge.tick_knowledge(&world, TEST_TICK, &mut rng);
        assert!(events.is_empty());
        assert_eq!(knowledge.active_researchers, 0);
    }

    #[test]
    fn test_diffusion_zero_with_high_cultural_distance() {
        let mut receiver = WorldKnowledge::new();
        let sender = WorldKnowledge::new();

        // Cultural distance = 1.0 -> zero absorption
        receiver.receive_diffusion(&sender, 1.0, 1.0);
        assert_eq!(receiver.diffusion_received, 0.0);
    }
}
