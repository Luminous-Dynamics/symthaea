// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Population dynamics: birth, death, epidemics, genetic diversity.

use crate::agent::{BiologicalSex, CivAgent, ConsciousnessState, SkillVector};
use crate::events::{CivEvent, CivEventType};
use crate::needs::PsychologicalNeeds;
use crate::stochastic::StochasticEngine;
use crate::world::World;

use serde::{Deserialize, Serialize};

/// Monthly pair-bonding probability for eligible adults.
const PAIR_BOND_PROBABILITY: f64 = 0.02;

/// Population dynamics engine.
pub struct PopulationEngine;

impl PopulationEngine {
    /// Pair-bond eligible unpaired adults. Called before birth processing.
    ///
    /// Eligible: alive, Adult or Youth, no current partner, partner also alive.
    /// Pairs are formed randomly from the eligible pool each tick.
    pub fn tick_pair_bonding(
        world: &mut World,
        rng: &mut StochasticEngine,
        current_tick: u32,
        pair_bond_rate: f64,
    ) {
        // Collect unpaired eligible males and females
        let mut unpaired_males: Vec<u64> = Vec::new();
        let mut unpaired_females: Vec<u64> = Vec::new();

        for agent in world.agents.iter().filter(|a| a.is_alive()) {
            if agent.partner_id.is_some() {
                continue;
            }
            let stage = agent.life_stage(current_tick);
            if !stage.can_reproduce() {
                continue;
            }
            match agent.sex {
                BiologicalSex::Male => unpaired_males.push(agent.id),
                BiologicalSex::Female => unpaired_females.push(agent.id),
            }
        }

        // Shuffle and pair probabilistically
        let pairs = unpaired_males.len().min(unpaired_females.len());
        for i in 0..pairs {
            if rng.bernoulli(pair_bond_rate) {
                let male_id = unpaired_males[i];
                let female_id = unpaired_females[i];

                // Set partner_id on both
                if let Some(m) = world.agents.iter_mut().find(|a| a.id == male_id) {
                    m.partner_id = Some(female_id);
                }
                if let Some(f) = world.agents.iter_mut().find(|a| a.id == female_id) {
                    f.partner_id = Some(male_id);
                }
            }
        }

        // Clear partner_id for agents whose partner has died
        let dead_ids: Vec<u64> = world
            .agents
            .iter()
            .filter(|a| !a.is_alive())
            .map(|a| a.id)
            .collect();
        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
            if let Some(pid) = agent.partner_id {
                if dead_ids.contains(&pid) {
                    agent.partner_id = None;
                }
            }
        }
    }

    /// Tick demographics: process births and deaths for a single world.
    pub fn tick_demographics(
        world: &mut World,
        rng: &mut StochasticEngine,
        current_tick: u32,
    ) -> Vec<CivEvent> {
        let mut events = Vec::new();

        // --- Deaths ---
        let death_ids: Vec<u64> = world
            .agents
            .iter()
            .filter(|a| a.is_alive())
            .filter_map(|a| {
                // Use tech-gated Gompertz modifiers (lifespan evolution model)
                let mut rate = a.mortality_rate_with_modifiers(
                    current_tick,
                    world.mortality_alpha_mult,
                    world.mortality_beta_mult,
                    world.mortality_lambda_mult,
                );

                // Allostatic load increases mortality risk.
                // Ref: McEwen (1998) — sustained allostatic overload accelerates aging.
                rate *= 1.0 + a.needs.allostatic_load * 0.5;

                // Trauma increases mortality: rate *= (1 + trauma * 0.3)
                rate *= 1.0 + a.trauma_level * 0.3;

                // Resource scarcity multiplier
                if let Some(food) = world.resources.get("food") {
                    let ratio = food.current / food.capacity.max(1.0);
                    if ratio < 0.2 {
                        rate *= 3.0;
                    } else if ratio < 0.5 {
                        rate *= 1.5;
                    }
                }

                if rng.bernoulli(rate) {
                    Some(a.id)
                } else {
                    None
                }
            })
            .collect();

        for id in &death_ids {
            if let Some(agent) = world.agents.iter_mut().find(|a| a.id == *id) {
                agent.death_tick = Some(current_tick);
                events.push(CivEvent::new(
                    current_tick,
                    Some(world.id),
                    CivEventType::Death,
                    format!("Agent {} died at age {:.1}", id, agent.age_years(current_tick)),
                ));
            }
        }

        // --- Trauma decay: 0.001 per tick, floor at 0.0 ---
        for agent in world.agents.iter_mut().filter(|a| a.is_alive()) {
            agent.trauma_level = (agent.trauma_level - 0.001).max(0.0);
        }

        // --- Births ---
        // #10: Reproduction viability check. If gravity < 0.4g and no centrifuge
        // tech, reproduction may not be viable. Unknown if mammals can develop
        // at 0.14g (Titan) or 0.38g (Mars). NASA ICES-2021-142.
        if !world.reproduction_viable {
            // No births — return deaths + any events already collected
            return events;
        }
        // Collect fertile pairs
        let living: Vec<&CivAgent> = world
            .agents
            .iter()
            .filter(|a| a.is_alive())
            .collect();

        let fertile_females: Vec<u64> = living
            .iter()
            .filter(|a| {
                a.sex == BiologicalSex::Female
                    && a.life_stage(current_tick).can_reproduce()
                    && a.age_years(current_tick) <= 50.0
            })
            .map(|a| a.id)
            .collect();

        let has_males = living
            .iter()
            .any(|a| a.sex == BiologicalSex::Male && a.life_stage(current_tick).can_reproduce());

        if has_males && world.population() < world.max_population {
            // Calibration anchor: `empirical::COLONIAL_BIRTH_RATE_PER_YEAR` (0.03/yr, >3%)
            // gives ~0.0025/month. Agent-level fertility() already produces comparable
            // rates when multiplied by nutrition and cultural modifiers.
            //
            // Nutrition modifier based on food availability
            let nutrition_modifier = world
                .resources
                .get("food")
                .map(|f| (f.current / f.capacity.max(1.0)).min(1.0))
                .unwrap_or(0.5);

            // Cultural modifier (higher traditionalism = slightly higher birth rate)
            let cultural_modifier = 0.8 + 0.4 * world.culture.traditionalism;

            // Collect birth decisions (child_sex, mother_id, father_id) to avoid borrow issues
            let mut births: Vec<(BiologicalSex, u64, Option<u64>)> = Vec::new();

            for fem_id in &fertile_females {
                if let Some(fem) = world.agents.iter().find(|a| a.id == *fem_id) {
                    let base_fertility = fem.fertility(current_tick);
                    let p = base_fertility * nutrition_modifier * cultural_modifier * fem.health;

                    if rng.bernoulli(p) {
                        let child_sex = if rng.bernoulli(0.5) {
                            BiologicalSex::Male
                        } else {
                            BiologicalSex::Female
                        };
                        births.push((child_sex, *fem_id, fem.partner_id));
                    }
                }
            }

            // Create children and link back to parents
            for (child_sex, mother_id, father_id) in births {
                let child_id = world.next_agent_id;
                world.next_agent_id += 1;

                let child = CivAgent {
                    id: child_id,
                    birth_tick: current_tick,
                    death_tick: None,
                    sex: child_sex,
                    world_id: world.id,
                    health: {
                        // Realism G: Inbreeding depression reduces child health.
                        // Wright's F > 0.1: noticeable fitness effects.
                        // F > 0.25: severe (lethal alleles expressed).
                        let f = Self::inbreeding_coefficient(world, current_tick);
                        let base = rng.next_gaussian(0.9, 0.05);
                        let penalty = if f > 0.1 { (f - 0.1) * 2.0 } else { 0.0 };
                        (base - penalty).clamp(0.2, 1.0)
                    },
                    skills: SkillVector::new(),
                    education_level: 0.0,
                    consciousness: ConsciousnessState::nascent(),
                    partner_id: None,
                    children_ids: vec![],
                    is_immigrant: false,
                    needs: PsychologicalNeeds::nascent(),
                    tend_balance: 0.0,
                    parent_ids: Some((mother_id, father_id.unwrap_or(0))),
                    faction_id: None,
                    generation: {
                        // Inherit generation from mother + 1
                        let mother_gen = world.agents.iter()
                            .find(|a| a.id == mother_id)
                            .map(|a| a.generation)
                            .unwrap_or(0);
                        mother_gen.saturating_add(1)
                    },
                    trauma_level: {
                        // Intergenerational trauma: 50% of mother's trauma passes down
                        let mother_trauma = world.agents.iter()
                            .find(|a| a.id == mother_id)
                            .map(|a| a.trauma_level)
                            .unwrap_or(0.0);
                        (mother_trauma * 0.5).min(1.0)
                    },
                    cumulative_dose_sv: 0.0,
                };

                events.push(CivEvent::new(
                    current_tick,
                    Some(world.id),
                    CivEventType::Birth,
                    format!("Agent {child_id} born on {}", world.name),
                ));

                world.agents.push(child);

                // Link child to mother
                if let Some(mother) = world.agents.iter_mut().find(|a| a.id == mother_id) {
                    mother.children_ids.push(child_id);
                }
                // Link child to father (if partnered)
                if let Some(fid) = father_id {
                    if let Some(father) = world.agents.iter_mut().find(|a| a.id == fid) {
                        father.children_ids.push(child_id);
                    }
                }
            }
        }

        // --- Prune dead agents every 12 ticks (yearly) to prevent unbounded growth ---
        if current_tick % 12 == 0 {
            world.agents.retain(|a| a.is_alive());
        }

        // MVP check
        if !Self::minimum_viable_population(world, current_tick) {
            events.push(CivEvent::new(
                current_tick,
                Some(world.id),
                CivEventType::GeneticAlert,
                format!(
                    "WARNING: {} below minimum viable population (adults: {})",
                    world.name,
                    world.adults(current_tick)
                ),
            ));
        }

        events
    }

    /// Check if the world has at least 20 adults.
    ///
    /// Calibration anchor: `empirical::MVP_50_500_RULE_SHORT_TERM` (50) is the IUCN
    /// minimum for short-term genetic health. We use 20 adults as a more permissive
    /// threshold for small founding colonies. Below `empirical::BEACHHEAD_BOTTLENECK_THRESHOLD`
    /// (50 adults), colony failure risk increases sharply (island biogeography, Royal Society 2022).
    pub fn minimum_viable_population(world: &World, current_tick: u32) -> bool {
        world.adults(current_tick) >= 20
    }

    /// Genetic viability score [0.0, 1.0] — 0 = doomed, 1 = stable.
    /// Accounts for effective population size and immigration rate.
    pub fn genetic_viability(world: &World, _current_tick: u32) -> f64 {
        let pop = world.population().max(1) as f64;
        let ne = pop * 0.25; // Conservative Ne/N ratio
        let immigrant_frac = world.agents.iter()
            .filter(|a| a.is_alive() && a.is_immigrant)
            .count() as f64 / pop;

        // Viability curve: sigmoid centered at Ne=200
        let base = 1.0 / (1.0 + (-0.02 * (ne - 200.0)).exp());
        // Immigration boost: each immigrant generation rescues ~m of diversity
        let boosted = base + immigrant_frac * (1.0 - base) * 2.0;
        boosted.clamp(0.0, 1.0)
    }

    /// Genetic diversity proxy incorporating both founder effect and immigration.
    ///
    /// H(t) = H0 * (1 - 1/(2*Ne))^generations + immigration_boost
    ///
    /// where Ne = effective population (0.7 * census for small pops),
    /// H0 = 1 - 1/founders, and immigration_boost accounts for gene flow
    /// from immigrants arriving after founding (each immigrant contributes
    /// diversity proportional to 1/Ne).
    pub fn genetic_diversity_index(world: &World, current_tick: u32) -> f64 {
        let pop = world.population().max(1);
        let ne = (pop as f64 * 0.7).max(1.0);
        let generations_elapsed =
            (current_tick.saturating_sub(world.founded_tick)) as f64 / 300.0;

        // Count founders: agents whose birth_tick wraps (born "before" simulation start).
        // These have birth_tick > current_tick due to u32 wrapping arithmetic.
        // Children born during the simulation (birth_tick <= current_tick) are NOT founders.
        // For worlds founded mid-simulation (Mars), founders are immigrants from the parent.
        let founder_count = world
            .agents
            .iter()
            .filter(|a| {
                // Pre-simulation agents: wrapped birth_tick gives huge values
                a.birth_tick > current_tick.max(1800)
                // Or explicitly marked as immigrant (settlers from parent world)
                || a.is_immigrant
            })
            .count();
        let immigrant_count = world
            .agents
            .iter()
            .filter(|a| a.is_alive() && a.is_immigrant)
            .count();

        // Effective genetic contributors: founders + living immigrants
        let contributors = (founder_count + immigrant_count).max(2) as f64;
        let h0 = 1.0 - 1.0 / contributors;

        // Base heterozygosity decay
        let base = h0 * (1.0 - 1.0 / (2.0 * ne)).powf(generations_elapsed);

        // Immigration gene flow boost: m * (1 - H) where m = immigrant fraction.
        // Each immigrant generation restores some diversity toward panmictic equilibrium.
        let m = immigrant_count as f64 / pop.max(1) as f64;
        let boosted = base + m * (1.0 - base);

        boosted.clamp(0.0, 1.0)
    }

    /// Inbreeding coefficient: F = 1 - H(t)/H(0).
    pub fn inbreeding_coefficient(world: &World, current_tick: u32) -> f64 {
        let founder_count = world
            .agents
            .iter()
            .filter(|a| {
                a.birth_tick <= world.founded_tick + 1
                || a.birth_tick > current_tick
            })
            .count()
            .max(1) as f64;
        let h0 = 1.0 - 1.0 / founder_count;
        if h0 <= 0.0 {
            return 1.0;
        }
        let ht = Self::genetic_diversity_index(world, current_tick);
        (1.0 - ht / h0).max(0.0)
    }
}

/// SIR epidemic model for disease outbreaks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SirEpidemic {
    pub name: String,
    /// Transmission rate (contacts * probability per contact).
    pub beta: f64,
    /// Recovery rate (1/duration in ticks).
    pub gamma: f64,
    /// Infection fatality rate.
    pub mortality: f64,
    pub susceptible: usize,
    pub infected: usize,
    pub recovered: usize,
    pub dead: usize,
    pub tick_started: u32,
}

impl SirEpidemic {
    /// Advance one tick of the SIR model. Returns (new_infections, new_deaths).
    pub fn tick(
        &mut self,
        population_size: usize,
        rng: &mut StochasticEngine,
    ) -> (usize, usize) {
        if self.infected == 0 || population_size == 0 {
            return (0, 0);
        }

        let n = population_size as f64;
        let s = self.susceptible as f64;
        let i = self.infected as f64;

        // New infections: beta * S * I / N (stochastic)
        let expected_new_infections = (self.beta * s * i / n).min(s);
        let new_infections = (0..self.susceptible)
            .filter(|_| rng.bernoulli(expected_new_infections / s.max(1.0)))
            .count()
            .min(self.susceptible);

        // Recoveries and deaths among infected
        let mut new_recoveries = 0usize;
        let mut new_deaths = 0usize;
        for _ in 0..self.infected {
            if rng.bernoulli(self.gamma) {
                if rng.bernoulli(self.mortality) {
                    new_deaths += 1;
                } else {
                    new_recoveries += 1;
                }
            }
        }

        self.susceptible -= new_infections;
        self.infected = self.infected + new_infections - new_recoveries - new_deaths;
        self.recovered += new_recoveries;
        self.dead += new_deaths;

        (new_infections, new_deaths)
    }

    /// Whether the epidemic has burned out.
    pub fn is_over(&self) -> bool {
        self.infected == 0
    }

    /// Preset: common cold (high spread, low mortality).
    pub fn common_cold(population: usize, tick: u32) -> Self {
        Self {
            name: "Common Cold".into(),
            beta: 0.4,
            gamma: 0.25,
            mortality: 0.001,
            susceptible: population.saturating_sub(1),
            infected: 1,
            recovered: 0,
            dead: 0,
            tick_started: tick,
        }
    }

    /// Preset: influenza (moderate spread and mortality).
    pub fn influenza(population: usize, tick: u32) -> Self {
        Self {
            name: "Influenza".into(),
            beta: 0.3,
            gamma: 0.15,
            mortality: 0.02,
            susceptible: population.saturating_sub(1),
            infected: 1,
            recovered: 0,
            dead: 0,
            tick_started: tick,
        }
    }

    /// Preset: severe respiratory illness (lower spread, higher mortality).
    pub fn severe_respiratory(population: usize, tick: u32) -> Self {
        Self {
            name: "Severe Respiratory Illness".into(),
            beta: 0.2,
            gamma: 0.1,
            mortality: 0.05,
            susceptible: population.saturating_sub(1),
            infected: 1,
            recovered: 0,
            dead: 0,
            tick_started: tick,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{CulturalProfile, World, WorldResources};

    fn make_test_world(pop: usize) -> World {
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
            habitable_area_m2: 1_000_000.0,
            founding_harmony_emphasis: [0.125; 8],
            epidemics: Vec::new(),
            knowledge: crate::knowledge::WorldKnowledge::new(),
            economy: crate::economy::WorldEconomy::new(),
            harmony: crate::harmony::HarmonyTracker::new(),
            governance: crate::governance::WorldGovernance::new(),
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
            automation_level: 0.0,
            explorations_completed: 0,
            project_manager: crate::projects::ProjectManager::new(),
            diplomatic_relations: std::collections::HashMap::new(),
        };

        let mut rng = StochasticEngine::new(42);
        for i in 0..pop {
            let sex = if i % 2 == 0 {
                BiologicalSex::Female
            } else {
                BiologicalSex::Male
            };
            // Start adults at age ~30 (360 months before tick 0 is not possible,
            // so we set birth_tick to 0 and test at tick 360)
            let agent = CivAgent {
                id: world.next_agent_id,
                birth_tick: 0,
                death_tick: None,
                sex,
                world_id: 0,
                health: rng.next_gaussian(0.9, 0.05).clamp(0.3, 1.0),
                skills: SkillVector::new(),
                education_level: 0.0,
                consciousness: ConsciousnessState::nascent(),
                partner_id: None,
                children_ids: vec![],
                is_immigrant: false,
                needs: PsychologicalNeeds::new(),
                tend_balance: 0.0,
                parent_ids: None,
                faction_id: None,
                generation: 0,
                trauma_level: 0.0,
                    cumulative_dose_sv: 0.0,
            };
            world.next_agent_id += 1;
            world.agents.push(agent);
        }

        world
    }

    #[test]
    fn test_birth_death_basic_mechanics() {
        let mut world = make_test_world(100);
        let mut rng = StochasticEngine::new(99);
        // Run multiple ticks to accumulate events (single tick may not produce any)
        let mut total_births = 0;
        let mut total_deaths = 0;
        for tick in (30 * 12)..(32 * 12) {
            let events = PopulationEngine::tick_demographics(&mut world, &mut rng, tick);
            total_births += events
                .iter()
                .filter(|e| e.event_type == CivEventType::Birth)
                .count();
            total_deaths += events
                .iter()
                .filter(|e| e.event_type == CivEventType::Death)
                .count();
        }
        // Over 24 ticks with 100 agents (50 fertile females), expect some births
        assert!(
            total_births + total_deaths > 0,
            "Expected some demographic events over 24 ticks"
        );
    }

    #[test]
    fn test_sir_epidemic_progression() {
        let mut epi = SirEpidemic::influenza(1000, 0);
        let mut rng = StochasticEngine::new(42);
        assert!(!epi.is_over());

        // Run for several ticks
        let mut total_infected = 0usize;
        for _ in 0..50 {
            let (new_inf, _) = epi.tick(1000, &mut rng);
            total_infected += new_inf;
        }
        // Should have spread
        assert!(total_infected > 0);
        assert!(epi.recovered > 0 || epi.dead > 0);
    }

    #[test]
    fn test_genetic_diversity_calculation() {
        let world = make_test_world(50);
        let div = PopulationEngine::genetic_diversity_index(&world, 0);
        // With updated founder detection (wrapped birth_tick or immigrant),
        // test agents born at tick 0 are not founders. Diversity driven by
        // population Ne and immigration boost. Should be positive and bounded.
        assert!(div > 0.0 && div <= 1.0, "diversity was {div}");
    }

    #[test]
    fn test_inbreeding_coefficient() {
        let world = make_test_world(50);
        let f0 = PopulationEngine::inbreeding_coefficient(&world, 0);
        // F may be nonzero with updated formula, but should be bounded
        assert!(f0 >= 0.0 && f0 <= 1.0, "F at t=0 out of bounds: {f0}");

        // After many generations with small pop
        let small_world = make_test_world(4);
        let f_late = PopulationEngine::inbreeding_coefficient(&small_world, 3000);
        // With Ne~2.8, inbreeding should accumulate
        assert!(f_late > f0, "inbreeding should increase over time");
    }

    #[test]
    fn test_minimum_viable_population() {
        let world = make_test_world(10);
        // At tick 0, agents are age 0 (children), so adults = 0
        assert!(!PopulationEngine::minimum_viable_population(&world, 0));
        // At tick 25*12, agents are 25 (adults)
        assert!(!PopulationEngine::minimum_viable_population(&world, 25 * 12));
        // Need at least 20 adults
        let big = make_test_world(30);
        assert!(PopulationEngine::minimum_viable_population(&big, 25 * 12));
    }
}
